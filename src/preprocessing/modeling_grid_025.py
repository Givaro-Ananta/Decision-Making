"""Exact 0.25-degree modeling cells and land-area overlap with verified base cells."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from rasterio.warp import transform_geom
from shapely import from_wkb, to_wkb
from shapely.geometry import box, mapping, shape

RESOLUTION = 0.25
WEST = 94.95
NORTH = 6.05


def cell_id(row, col, resolution=RESOLUTION):
    return f'SUM{int(round(resolution*100)):03d}_R{int(row):03d}_C{int(col):03d}'


def point_indices(longitude, latitude, resolution=RESOLUTION):
    """North/west included, south/east excluded; stable at decimal FIRMS coordinates."""
    import numpy as np
    col = np.floor(np.round((np.asarray(longitude)-WEST)/resolution, 10)).astype('int32')
    row = np.floor(np.round((NORTH-np.asarray(latitude))/resolution, 10)).astype('int32')
    return row, col


def area_m2(geometry):
    return shape(transform_geom('EPSG:4326','EPSG:6933',mapping(geometry))).area


def build_grid_overlap(base_grid: pd.DataFrame, domain, resolution=RESOLUTION):
    """Build target rectangles, their Sumatra support and exact intersecting base-cell weights."""
    target_cells={}
    overlaps=[]
    for item in base_grid.itertuples(index=False):
        base=from_wkb(item.geometry)
        support=base if domain.covers(base) else base.intersection(domain)
        if support.is_empty or support.area<=1e-12:
            continue
        xmin,ymin,xmax,ymax=base.bounds
        r0=math.floor((NORTH-ymax+1e-10)/resolution)
        r1=math.floor((NORTH-ymin-1e-10)/resolution)
        c0=math.floor((xmin-WEST+1e-10)/resolution)
        c1=math.floor((xmax-WEST-1e-10)/resolution)
        for row in range(r0,r1+1):
            for col in range(c0,c1+1):
                target=box(WEST+col*resolution,NORTH-(row+1)*resolution,
                           WEST+(col+1)*resolution,NORTH-row*resolution)
                overlap=support.intersection(target)
                if overlap.is_empty or overlap.area<=1e-12:
                    continue
                gid=cell_id(row,col,resolution)
                overlaps.append({'base_grid_id':item.grid_id,'grid_id':gid,'overlap_area_m2':area_m2(overlap)})
                if gid not in target_cells:
                    target_support=target.intersection(domain)
                    target_cells[gid]={
                        'grid_id':gid,'geometry':to_wkb(target),'study_geometry':to_wkb(target_support),
                        'centroid_lat':NORTH-(row+.5)*resolution,
                        'centroid_lon':WEST+(col+.5)*resolution,
                        'row_index':row,'col_index':col,
                        'geometric_land_area_m2':area_m2(target_support),
                    }
    grid=pd.DataFrame(target_cells.values()).sort_values(['row_index','col_index']).reset_index(drop=True)
    weights=pd.DataFrame(overlaps)
    areas=weights.groupby('grid_id')['overlap_area_m2'].sum().rename('weighted_land_area_m2')
    grid=grid.merge(areas,on='grid_id',validate='one_to_one')
    error=(grid['weighted_land_area_m2']-grid['geometric_land_area_m2']).abs()/grid['geometric_land_area_m2']
    # Reprojected polygon chords can differ very slightly when split at extra vertices.
    if error.max()>1e-3:
        raise ValueError('Land overlap weights fail target area conservation')
    return grid,weights
