#!/usr/bin/env python3
"""Create an exact 0.25-degree panel; never modify the verified source panel."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from shapely import prepare
from shapely.geometry import box
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.config import START_DATE, END_DATE, FIRMS_DIR, ERA5_DIR, GRID_FILE
from src.preprocessing.canonical_grid import load_mainland_sumatra_geometry, GRID_EDGE_BOUNDS
from src.preprocessing.modeling_grid_025 import build_grid_overlap, point_indices, cell_id
from src.preprocessing.preprocess_firms import FIRMS_FILES, filter_sumatra
from src.preprocessing.preprocess_era5 import _time_by_band, _daily_reduce

def literal(value):
    return "'"+str(value).replace("'","''")+"'"

def main():
    import duckdb
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=ROOT/'data/processed/sumatra_grid_day.parquet')
    ap.add_argument('--output',type=Path,default=ROOT/'data/processed/modeling_025')
    ap.add_argument('--resolution',type=float,choices=[.25,.5],default=.25)
    ap.add_argument('--humidity-cache',type=Path)
    args=ap.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=True)
    if args.input.resolve()==(out/'grid_day.parquet').resolve(): raise ValueError('Cannot overwrite source')
    before=(args.input.stat().st_size,args.input.stat().st_mtime_ns)
    base=pd.read_parquet(GRID_FILE)
    domain=load_mainland_sumatra_geometry(ROOT).intersection(box(*GRID_EDGE_BOUNDS)); prepare(domain)
    grid,weights=build_grid_overlap(base,domain,args.resolution)
    grid.to_parquet(out/'grid.parquet',index=False); weights.to_parquet(out/'base_grid_overlap.parquet',index=False)
    print(f'Exact grid: {len(grid)} units; {len(weights)} overlaps',flush=True)
    active_path=out/'firms_daily.parquet'
    if not active_path.exists():
        pieces=[]
        for name in FIRMS_FILES:
            for chunk in pd.read_csv(FIRMS_DIR/name,usecols=['latitude','longitude','acq_date','confidence','frp','daynight'],chunksize=200000):
                data=filter_sumatra(chunk,domain)
                if data.empty: continue
                if not np.isfinite(data.frp).all() or (data.frp<0).any(): raise ValueError('Invalid FRP')
                row,col=point_indices(data.longitude,data.latitude,args.resolution)
                data=data.assign(grid_id=[cell_id(r,c,args.resolution) for r,c in zip(row,col)],date=pd.to_datetime(data.acq_date))
                if not data.grid_id.isin(grid.grid_id).all(): raise ValueError('Point outside modeling cells')
                pieces.append(data[['grid_id','date','frp']])
            print(f'FIRMS read: {name}',flush=True)
        points=pd.concat(pieces,ignore_index=True)
        active=points.groupby(['grid_id','date']).agg(hotspot_count=('frp','size'),frp_sum=('frp','sum'),frp_mean=('frp','mean'),frp_max=('frp','max')).reset_index()
        active.to_parquet(active_path,index=False)
        del points,pieces,active
    rhdir=args.humidity_cache or out/'humidity_monthly'; rhdir.mkdir(exist_ok=True)
    rows=base.row_index.to_numpy(); cols=base.col_index.to_numpy()
    for path in sorted(ERA5_DIR.glob('era5_land_*.nc')):
        dest=rhdir/(path.stem+'.parquet')
        if dest.exists(): continue
        frames=[]
        with rasterio.open(f'NETCDF:{path}:t2m') as t, rasterio.open(f'NETCDF:{path}:d2m') as d:
            times=_time_by_band(t)
            if times!=_time_by_band(d) or t.transform!=d.transform: raise ValueError('Humidity alignment failure')
            for day,bands in sorted(times.items()):
                if not pd.Timestamp(START_DATE)<=day<=pd.Timestamp(END_DATE): continue
                if len(bands)!=4: values=np.full(len(base),np.nan)
                else:
                    temp=np.stack([t.read(b,masked=True).filled(np.nan) for b in bands])-273.15
                    dew=np.stack([d.read(b,masked=True).filled(np.nan) for b in bands])-273.15
                    rh=np.clip(100*np.exp(17.625*dew/(243.04+dew)-17.625*temp/(243.04+temp)),0,100)
                    values=_daily_reduce(rh,'mean')[rows,cols]
                frames.append(pd.DataFrame({'grid_id':base.grid_id,'date':day,'rh':values}))
        if frames: pd.concat(frames,ignore_index=True).to_parquet(dest,index=False)
        print(f'Humidity corrected: {path.stem}',flush=True)
    con=duckdb.connect(str(out/'preparation.duckdb'))
    con.execute("SET memory_limit='2GB'; SET threads=1")
    con.execute(f'CREATE OR REPLACE VIEW source AS SELECT * FROM read_parquet({literal(args.input)})')
    con.execute(f'CREATE OR REPLACE VIEW weights AS SELECT * FROM read_parquet({literal(out/"base_grid_overlap.parquet")})')
    con.execute(f'CREATE OR REPLACE VIEW grids AS SELECT * FROM read_parquet({literal(out/"grid.parquet")})')
    con.execute(f'CREATE OR REPLACE VIEW rh AS SELECT * FROM read_parquet({literal(rhdir/"*.parquet")})')
    con.execute(f'CREATE OR REPLACE VIEW active AS SELECT * FROM read_parquet({literal(active_path)})')
    features={'temperature_max_c_t':'s.temperature_t','relative_humidity_mean_t':'h.rh','wind_speed_mean_t':'s.wind_speed_t','rainfall_mm_t':'s.rain_t'}
    reductions=[]
    for name,expression in features.items():
        reductions += [f'sum({expression}*w.overlap_area_m2) FILTER(WHERE isfinite({expression}))/sum(w.overlap_area_m2) FILTER(WHERE isfinite({expression})) AS {name}',f'coalesce(sum(w.overlap_area_m2) FILTER(WHERE isfinite({expression})),0)/sum(w.overlap_area_m2) AS {name}_coverage']
    con.execute('DROP TABLE IF EXISTS panel')
    for i,month in enumerate(pd.period_range(START_DATE,END_DATE,freq='M')):
        start=max(month.start_time,pd.Timestamp(START_DATE)).date(); end=min(month.end_time,pd.Timestamp(END_DATE)).date()
        query=f'''WITH cov AS (SELECT w.grid_id,s.date,{','.join(reductions)} FROM source s JOIN weights w ON s.grid_id=w.base_grid_id LEFT JOIN rh h ON s.grid_id=h.grid_id AND s.date=h.date WHERE s.date BETWEEN DATE '{start}' AND DATE '{end}' GROUP BY w.grid_id,s.date), dates AS (SELECT CAST(d AS DATE) AS date FROM generate_series(DATE '{start}',DATE '{end}',INTERVAL 1 DAY) t(d)) SELECT g.grid_id,d.date,coalesce(a.hotspot_count,0)::BIGINT hotspot_count,coalesce(a.frp_sum,0)::DOUBLE frp_sum,a.frp_mean,a.frp_max,{','.join('c.'+n+',c.'+n+'_coverage' for n in features)},coalesce({' AND '.join('c.'+n+'_coverage>=1-1e-8 AND isfinite(c.'+n+')' for n in features)},false) AS predictors_available FROM grids g CROSS JOIN dates d LEFT JOIN cov c ON c.grid_id=g.grid_id AND c.date=d.date LEFT JOIN active a ON a.grid_id=g.grid_id AND a.date=d.date'''
        con.execute(('CREATE TABLE panel AS ' if i==0 else 'INSERT INTO panel ')+query)
        print(f'Panel month: {month}',flush=True)
    stats=con.execute('SELECT count(*),count(DISTINCT grid_id),count(DISTINCT date),sum(hotspot_count),sum(frp_sum),count(*) FILTER(WHERE predictors_available) FROM panel').fetchone()
    expected=len(grid)*len(pd.date_range(START_DATE,END_DATE))
    assert stats[0]==expected
    assert con.execute('SELECT count(*) FROM (SELECT grid_id,date FROM panel GROUP BY ALL HAVING count(*)<>1)').fetchone()[0]==0
    gold=con.execute('SELECT sum(hotspot_count),sum(frp_sum) FROM source').fetchone()
    assert stats[3]==gold[0] and np.isclose(stats[4],gold[1],rtol=1e-10)
    con.execute(f'COPY (SELECT * FROM panel ORDER BY grid_id,date) TO {literal(out/"grid_day.parquet")} (FORMAT PARQUET,COMPRESSION ZSTD)')
    assert before==(args.input.stat().st_size,args.input.stat().st_mtime_ns)
    report=dict(zip(['rows','grids','dates','hotspots','frp_sum','predictor_available_rows'],stats))
    report.update(start_date=START_DATE,end_date=END_DATE,expected_rows=expected,source_unchanged=True,source_stat=before)
    (out/'preparation_audit.json').write_text(json.dumps(report,indent=2))
    con.close(); print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__': main()
