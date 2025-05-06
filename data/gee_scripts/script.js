// define region of interest

var gaul = ee.FeatureCollection("FAO/GAUL/2015/level2");
var alaska_roi = gaul.filter(ee.Filter.eq('ADM1_NAME','Alaska'));
Map.addLayer(alaska_roi);

// longitude bounds
var minLong = -168;
var maxLong = -141;

// split region into bands
var band1 = ee.Geometry.Polygon([[[minLong, 61.2755545], [minLong, 59.2632802], [maxLong, 59.2632802], [maxLong, 61.2755545]]]);
band1 = band1.intersection(alaska_roi.geometry(), ee.ErrorMargin(1));
var band2 = ee.Geometry.Polygon([[[minLong, 63.2878288], [minLong, 61.2755545], [maxLong, 61.2755545], [maxLong, 63.2878288]]]);
band2 = band2.intersection(alaska_roi.geometry(), ee.ErrorMargin(1));
var band3 = ee.Geometry.Polygon([[[minLong, 65.3001031], [minLong, 63.2878288], [maxLong, 63.2878288], [maxLong, 65.3001031]]]);
band3 = band3.intersection(alaska_roi.geometry(), ee.ErrorMargin(1));
var band4 = ee.Geometry.Polygon([[[minLong, 67.3123774], [minLong, 65.3001031], [maxLong, 65.3001031], [maxLong, 67.3123774]]]);
band4 = band4.intersection(alaska_roi.geometry(), ee.ErrorMargin(1));
var band5 = ee.Geometry.Polygon([[[minLong, 69.3246517], [minLong, 67.3123774], [maxLong, 67.3123774], [maxLong, 69.3246517]]]);
band5 = band5.intersection(alaska_roi.geometry(), ee.ErrorMargin(1));
var band6 = ee.Geometry.Polygon([[[minLong, 71.336926], [minLong, 69.3246517], [maxLong, 69.3246517], [maxLong, 71.336926]]]);
band6 = band6.intersection(alaska_roi.geometry(), ee.ErrorMargin(1));

// obtain data from era5 dataset for each year 

var features = [
    'lake_bottom_temperature', 
        'lake_ice_depth',
        'lake_ice_temperature',
        'lake_mix_layer_depth',
        'lake_mix_layer_temperature',
        'lake_shape_factor',
        'lake_total_layer_temperature',
        'leaf_area_index_high_vegetation',
        'leaf_area_index_low_vegetation',
        'snow_albedo',
        'snow_cover',
        'snow_density',
        'snow_depth',
        'snow_depth_water_equivalent',
        'snowfall_sum',
        'snowmelt_sum',
        'temperature_of_snow_layer',
        'soil_temperature_level_1',
        'soil_temperature_level_2',
        'soil_temperature_level_3',
        'soil_temperature_level_4',
        'volumetric_soil_water_layer_1',
        'volumetric_soil_water_layer_2',
        'volumetric_soil_water_layer_3',
        'volumetric_soil_water_layer_4',
        'forecast_albedo',
        'surface_latent_heat_flux_sum',
        'surface_net_solar_radiation_sum',
        'surface_net_thermal_radiation_sum', 'surface_sensible_heat_flux_sum',
        'surface_solar_radiation_downwards_sum',
        'surface_thermal_radiation_downwards_sum',
        'dewpoint_temperature_2m',
        'skin_temperature',
        'temperature_2m',
        'total_evaporation_sum',
        'total_precipitation_sum'
];

// information on this dataset can be found at https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_AGGR
var era5_land_data = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
    .filterDate('2001-01-01', '2023-12-31')
    .select(features);

// export data into google drive

//band1
var timeSeries_1 = ee.FeatureCollection(era5_land_data.map(function(image) {
    var values = image.reduceRegion({
        reducer: ee.Reducer.first(),
        geometry: band1,
        scale: 5000
    });

    var data = ee.Dictionary.fromLists(
        features,
        features.map(function(f) {
          return values.get(f);
        })
      );
    
    var feature_data = ee.Dictionary({'date': image.date().format('YYYY-MM-dd')})
    .combine(data);
    
    return ee.Feature(null, feature_data);
  }));
  
Export.table.toDrive({
collection: timeSeries_1,
description: 'band1_data',
fileFormat: 'CSV',
folder: 'bands', // create a folder in drive called 'bands'
});

//band2
var timeSeries_2 = ee.FeatureCollection(era5_land_data.map(function(image) {
    var values = image.reduceRegion({
        reducer: ee.Reducer.first(),
        geometry: band2,
        scale: 5000
    });

    var data = ee.Dictionary.fromLists(
        features,
        features.map(function(f) {
          return values.get(f);
        })
      );
    
    var feature_data = ee.Dictionary({'date': image.date().format('YYYY-MM-dd')})
    .combine(data);
    
    return ee.Feature(null, feature_data);
  }));
  
Export.table.toDrive({
collection: timeSeries_2,
description: 'band2_data',
fileFormat: 'CSV',
folder: 'bands', 
});

//band3
var timeSeries_3 = ee.FeatureCollection(era5_land_data.map(function(image) {
    var values = image.reduceRegion({
        reducer: ee.Reducer.first(),
        geometry: band3,
        scale: 5000
    });

    var data = ee.Dictionary.fromLists(
        features,
        features.map(function(f) {
          return values.get(f);
        })
      );
    
    var feature_data = ee.Dictionary({'date': image.date().format('YYYY-MM-dd')})
    .combine(data);
    
    return ee.Feature(null, feature_data);
  }));
  
Export.table.toDrive({
collection: timeSeries_3,
description: 'band3_data',
fileFormat: 'CSV',
folder: 'bands', 
});

//band 4
var timeSeries_4 = ee.FeatureCollection(era5_land_data.map(function(image) {
    var values = image.reduceRegion({
        reducer: ee.Reducer.first(),
        geometry: band4,
        scale: 5000
    });

    var data = ee.Dictionary.fromLists(
        features,
        features.map(function(f) {
          return values.get(f);
        })
      );
    
    var feature_data = ee.Dictionary({'date': image.date().format('YYYY-MM-dd')})
    .combine(data);
    
    return ee.Feature(null, feature_data);
  }));
  
Export.table.toDrive({
collection: timeSeries_4,
description: 'band4_data',
fileFormat: 'CSV',
folder: 'bands', 
});

// band5
var timeSeries_5 = ee.FeatureCollection(era5_land_data.map(function(image) {
    var values = image.reduceRegion({
        reducer: ee.Reducer.first(),
        geometry: band5,
        scale: 5000
    });

    var data = ee.Dictionary.fromLists(
        features,
        features.map(function(f) {
          return values.get(f);
        })
      );
    
    var feature_data = ee.Dictionary({'date': image.date().format('YYYY-MM-dd')})
    .combine(data);
    
    return ee.Feature(null, feature_data);
  }));
  
Export.table.toDrive({
collection: timeSeries_5,
description: 'band5_data',
fileFormat: 'CSV',
folder: 'bands', 
});

//band 6
var timeSeries_6 = ee.FeatureCollection(era5_land_data.map(function(image) {
    var values = image.reduceRegion({
        reducer: ee.Reducer.first(),
        geometry: band6,
        scale: 5000
    });

    var data = ee.Dictionary.fromLists(
        features,
        features.map(function(f) {
          return values.get(f);
        })
      );
    
    var feature_data = ee.Dictionary({'date': image.date().format('YYYY-MM-dd')})
    .combine(data);
    
    return ee.Feature(null, feature_data);
  }));
  
Export.table.toDrive({
collection: timeSeries_6,
description: 'band6_data',
fileFormat: 'CSV',
folder: 'bands', 
});