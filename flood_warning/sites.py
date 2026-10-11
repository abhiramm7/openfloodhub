"""Gauge registry, grouped into regions. Each region is its own deployment:
its own preds.json, its own view on the map. Models stay per gauge either way.

dmv   7 DC-area USGS gauges within ~20 km of downtown. Mix of urban
      (Anacostia, Rock Creek, Watts Branch), suburban (Difficult Run,
      NW Anacostia), and the Potomac mainstem at Little Falls. Drainage areas
      span 3.6 mi² (Watts Branch) to 11,560 mi² (Potomac at Little Falls) —
      the CNN gets a broad range of basin behaviors to learn from at this
      hourly cadence. Dropped from an earlier set: Potomac at Point of Rocks,
      Goose Creek, Catoctin Creek — all >40 km out of the DC core.

iowa  The Turkey River basin above Garber, NE Iowa (HUC-8 07060004): one
      subcatchment gauged at its outlet (Garber, 1,545 mi²) with six USGS
      gauges nested inside it — three upstream on the Turkey mainstem
      (Spillville -> Eldorado -> Elkader), the Volga River at Fayette and at
      Littleport (it joins the Turkey above Garber), and Roberts Creek, a karst
      tributary. No flood-control dams. The Iowa Flood Center also runs stream
      stage sensors and rain gauges in the basin; this registry only lists the
      USGS discharge gauges, since discharge is what the CNN trains on.
"""

REGIONS = {
    'dmv': {
        'name': 'Washington, DC',
        'model_id': 'dmv-cnn-12h',
        'marquee': '01646500',      # gauge the map opens on
    },
    'iowa': {
        'name': 'Turkey River, Iowa',
        'model_id': 'iowa-turkey-cnn-12h',
        'marquee': '05412500',
    },
}
DEFAULT_REGION = 'dmv'

SITES = [
    # ---- dmv ---------------------------------------------------------------
    {
        'id': '01646500', 'name': 'Potomac at Little Falls', 'short': 'Potomac DC',
        'region': 'dmv',
        'lat': 38.9498, 'lon': -77.1276,
        'drainage_sqmi': 11560,
        'kind': 'mainstem',
        'notes': 'The headline DC-region gauge. Drains 11,500 sq mi.',
    },
    {
        'id': '01648000', 'name': 'Rock Creek at Sherrill Dr', 'short': 'Rock Creek',
        'region': 'dmv',
        'lat': 38.9725, 'lon': -77.04,
        'drainage_sqmi': 62.2,
        'kind': 'urban',
        'notes': 'Rock Creek through NW DC — flashy urban watershed.',
    },
    {
        'id': '01651760', 'name': 'Anacostia at Kenilworth', 'short': 'Anacostia',
        'region': 'dmv',
        'lat': 38.9092, 'lon': -76.9553,
        'drainage_sqmi': 134,
        'kind': 'urban',
        'notes': 'Anacostia mainstem at NE DC.',
    },
    {
        'id': '01649500', 'name': 'NE Branch Anacostia at Riverdale', 'short': 'NE Anacostia',
        'region': 'dmv',
        'lat': 38.9603, 'lon': -76.926,
        'drainage_sqmi': 72.8,
        'kind': 'urban',
        'notes': 'Major Anacostia tributary, suburban PG County.',
    },
    {
        'id': '01650500', 'name': 'NW Branch Anacostia nr Colesville', 'short': 'NW Anacostia',
        'region': 'dmv',
        'lat': 39.0655, 'lon': -77.0294,
        'drainage_sqmi': 21.1,
        'kind': 'urban',
        'notes': 'NW Branch Anacostia in MoCo.',
    },
    {
        'id': '01651800', 'name': 'Watts Branch at DC', 'short': 'Watts Branch',
        'region': 'dmv',
        'lat': 38.9013, 'lon': -76.9433,
        'drainage_sqmi': 3.59,
        'kind': 'urban',
        'notes': 'Small urban tributary in SE DC — flash flood prone.',
    },
    {
        'id': '01646000', 'name': 'Difficult Run nr Great Falls', 'short': 'Difficult Run',
        'region': 'dmv',
        'lat': 38.9759, 'lon': -77.2458,
        'drainage_sqmi': 57.9,
        'kind': 'suburban',
        'notes': 'NoVa suburban watershed, flashy on storms.',
    },

    # ---- iowa: Turkey River above Garber -------------------------------------
    # Coordinates are the USGS NAD83 site locations.
    {
        'id': '05412500', 'name': 'Turkey River at Garber', 'short': 'Turkey Garber',
        'region': 'iowa',
        'lat': 42.7400, 'lon': -91.2618,
        'drainage_sqmi': 1545,
        'kind': 'mainstem',
        'notes': 'Subcatchment outlet, below the Volga confluence. NWS GRBI4.',
    },
    {
        'id': '05412020', 'name': 'Turkey River above French Hollow Cr at Elkader',
        'short': 'Turkey Elkader',
        'region': 'iowa',
        'lat': 42.8435, 'lon': -91.4013,
        'drainage_sqmi': 903,
        'kind': 'mainstem',
        'notes': 'Lower mainstem through Elkader. NWS ELKI4.',
    },
    {
        'id': '05411850', 'name': 'Turkey River near Eldorado', 'short': 'Turkey Eldorado',
        'region': 'iowa',
        'lat': 43.0542, 'lon': -91.8091,
        'drainage_sqmi': 641,
        'kind': 'mainstem',
        'notes': 'Mid-basin mainstem.',
    },
    {
        'id': '05411600', 'name': 'Turkey River at Spillville', 'short': 'Turkey Spillville',
        'region': 'iowa',
        'lat': 43.2073, 'lon': -91.9503,
        'drainage_sqmi': 177,
        'kind': 'agricultural',
        'notes': 'Upper Turkey headwaters. NWS SPLI4.',
    },
    {
        'id': '05412340', 'name': 'Volga River at Fayette', 'short': 'Volga Fayette',
        'region': 'iowa',
        'lat': 42.8439, 'lon': -91.8181,
        'drainage_sqmi': 130,
        'kind': 'agricultural',
        'notes': 'Upper Volga; USGS runs a rain gauge at the same site.',
    },
    {
        'id': '05412400', 'name': 'Volga River at Littleport', 'short': 'Volga Littleport',
        'region': 'iowa',
        'lat': 42.7539, 'lon': -91.3690,
        'drainage_sqmi': 348,
        'kind': 'agricultural',
        'notes': 'Lower Volga, above its Turkey confluence.',
    },
    {
        'id': '05412100', 'name': 'Roberts Creek above Saint Olaf', 'short': 'Roberts Creek',
        'region': 'iowa',
        'lat': 42.9303, 'lon': -91.3843,
        'drainage_sqmi': 70.7,
        'kind': 'karst',
        'notes': 'Karst tributary (Big Spring basin), long USGS research record.',
    },
]

BY_ID = {s['id']: s for s in SITES}


def sites_in(region: str | None = None) -> list[dict]:
    """Sites of one region, or every site when region is None."""
    if region is not None and region not in REGIONS:
        raise ValueError(f'unknown region {region!r} (known: {", ".join(REGIONS)})')
    return [s for s in SITES if region is None or s['region'] == region]
