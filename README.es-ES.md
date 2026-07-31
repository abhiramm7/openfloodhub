

# OpenFloodHub

Una alternativa de código abierto y autoalojable a [Google Flood Hub](https://sites.research.google/floods/).

Google Flood Hub ejecuta un modelo global: una sola red neuronal, entrenada en miles de cuencas, que predice el caudal en cualquier lugar con un único conjunto de pesos. OpenFloodHub hace lo contrario. Entrena un modelo pequeño y separado para cada estación hidrométrica, utilizando exclusivamente el historial de esa estación.

Esa compensación es la idea central del proyecto:

- Un modelo que solo necesita aprender una cuenca puede ser minúsculo (unos 50k parámetros), entrenarse en 90 segundos en la CPU de un portátil, y puedes inspeccionar realmente lo que aprendió para ese río específico.
- Un modelo global generaliza a lugares sin ninguna estación, algo que un modelo por sitio no puede hacer. Necesitas varios años de registros para cada estación antes de poder entrenarlo.

Por lo tanto, esto no es un sustituto directo para todo lo que hace Flood Hub. Es una apuesta diferente: para un conjunto conocido de estaciones que te interesen, un modelo local especializado es más fácil de ejecutar, más barato de volver a entrenar y más sencillo de analizar que un modelo grande que intenta cubrir todo el planeta.

El repositorio incluye un despliegue funcional para las estaciones alrededor de Washington, DC.

> **Advertencia.** Este es un prototipo de investigación. No lo uses para decidir si conducir por agua inundada. Las fuentes oficiales para eso son [NWS AHPS](https://water.weather.gov/ahps/) y [NOAA NWPS](https://water.noaa.gov/).

## El mapa

`web/` es una interfaz de mapa estática, estilizada como Flood Hub pero impulsada por los modelos por sitio. Sin paso de compilación, sin backend: un archivo HTML, un archivo JS, Leaflet y el `preds.json` que escribe el modelo.

![OpenFloodHub DC map](docs/screenshot.png)

Haz clic en una estación para abrir su panel. El gráfico muestra el caudal observado reciente (línea continua), la predicción de 12 horas del CNN local (línea discontinua) y el Modelo Nacional de Agua de NOAA para comparación (línea punteada). Cada estación tiene sus propios umbrales de Advertencia/Peligro/Extremo, derivados del historial de inundaciones de esa estación, por lo que la codificación de riesgo significa algo diferente para el Potomac que para un arroyo urbano de 4 millas cuadradas.

Para ejecutarlo localmente después de generar `preds.json` (ver abajo):

```bash
cp outputs/dmv-cnn-12h/preds.json web/preds.json
cd web && python3 -m http.server 8772      # open http://localhost:8772
```

## El modelo

`flood_warning/model.py`. CNN 1D de dos ramas: una corriente pasada de 4 canales (caudal, precipitación, temperatura y humedad del suelo para las últimas 24 horas) y una corriente futura de 1 canal (precipitación pronosticada para las próximas 12 horas). Cada una pasa por algunas capas Conv1D, se aplanan, se concatenan y se proyectan a una salida de 12 pasos. El 4º canal pasado es la humedad superficial del suelo de ERA5-Land: un indicador de humedad antecedente que indica al modelo qué tan saturada está la cuenca antes de una tormenta.

```
past   (4 ch x 24h)  ->  Conv1D x 3  ->  flatten
future (1 ch x 12h)  ->  Conv1D x 2  ->  flatten
                         concat -> FC -> 12-step forecast
```

Aproximadamente 50k parámetros. Se entrena por estación en 90 segundos en CPU.

NSE de prueba en datos retenidos (3 años de datos por hora, últimos 15% como prueba):

| Estación | Cuenca (mi²) | NSE de prueba | NSE a 12h |
| --- | ---: | ---: | ---: |
| Potomac en Little Falls (DC) | 11,560 | 0.977 | 0.945 |
| Anacostia en Kenilworth (DC) | 134 | 0.694 | 0.653 |
| NE Branch Anacostia (MD) | 73 | 0.436 | 0.175 |
| Rock Creek en Sherrill Dr (DC) | 62 | 0.414 | 0.144 |
| Difficult Run (VA) | 58 | 0.352 | 0.115 |
| NW Branch Anacostia (MD) | 21 | 0.211 | 0.061 |
| Watts Branch (DC) | 3.6 | 0.123 | 0.024 |

Las grandes estaciones en el cauce principal están básicamente resueltas a resolución horaria. Las pequeñas cuencas urbanas en la parte inferior de la tabla son difíciles y no hay forma real de evitarlo: Watts Branch son 3.6 mi² de pavimento, responde a la lluvia en minutos y un modelo por hora es la herramienta incorrecta. O aumenta la cadencia a 15 minutos o aliméntalo con precipitación NEXRAD en lugar de datos puntuales de ERA5.

## Umbrales

El NWS publica las categorías de inundación principalmente como altura en la estación (nivel en pies), algo a lo que un modelo de caudal no puede responder. Por eso, `thresholds.py` deriva umbrales de flujo (m³/s) a partir del historial de cada estación: toma la distribución de picos diarios a lo largo de la historia multianual y extrae cuantiles altos como sustitutos del período de retorno (aproximadamente 2, 5 y 10 años para Advertencia, Peligro y Extremo). Estos se almacenan en caché en `thresholds.json` y se adjuntan a cada predicción.

## Estructura

```
flood_warning/
├── sites.py            # the gauges + lat/lon/drainage
├── fetch.py            # USGS NWIS + Open-Meteo data fetcher
├── dataset.py          # windowing + scaler
├── model.py            # the CNN
├── train.py            # per-gauge training
├── predict.py          # live inference + 7-day backtest
├── thresholds.py       # per-gauge flood thresholds from the record
├── thresholds.json     # cached thresholds (committed)
├── noaa.py             # NOAA/NWS comparison overlays (NWM, QPF, MRMS)
├── checkpoints/        # pretrained .pt files (one per gauge)
└── requirements-ci.txt

web/                    # static map UI (index.html, app.js, preds.json)
```

## Configuración

Python 3.12. Obtén una clave API gratuita de USGS en [waterdata.usgs.gov](https://waterdata.usgs.gov) y colócala en `.env.local` como `USGS_API_KEY=...`.

```bash
uv venv --python 3.12 .venv
.venv/bin/pip install -r requirements.txt
```

Los datos en caché se guardan en `./data/` por defecto; sobrescríbelo con `$FLOOD_DATA_DIR`. La inferencia escribe en `./outputs/`.

## Entrenar desde cero

```bash
.venv/bin/python -m flood_warning.fetch          # ~3 years of hourly data
for gid in 01646500 01648000 01651760 01649500 01650500 01651800 01646000; do
  .venv/bin/python -m flood_warning.train "$gid"
done                                              # ~90s per gauge on CPU
.venv/bin/python -m flood_warning.thresholds      # compute flood thresholds
```

## Ejecutar inferencia

```bash
.venv/bin/python -m flood_warning.predict        # writes outputs/dmv-cnn-12h/preds.json
```

`preds.json` contiene, por estación: el pronóstico del CNN con 12 horas de antelación, una prueba retroactiva horaria de 7 días (cómo ha estado siguiendo el caudal observado el modelo últimamente), los umbrales de inundación de la estación y un conjunto de series de comparación de NOAA/NWS. Las series son solo de referencia. Se muestran junto con el pronóstico del CNN y nunca se retroalimentan al modelo:

| Campo | Origen | Descripción |
| --- | --- | --- |
| `noaa_nwm` | NWM corto plazo | Caudal fluvial próximas ~18h, por hora (m³/s) |
| `noaa_nwm_medium` | NWM mezcla mediano plazo | Caudal fluvial próximas ~10d, por hora (m³/s) |
| `noaa_nwm_analysis` | NWM análisis-asimilación | Caudal fluvial "observado" estimado más reciente (m³/s) |
| `noaa_qpf` | Pronóstico por punto del NWS | Lluvia pronosticada, por hora (mm) |
| `noaa_mrms_precip` | QPE radar MRMS (vía IEM) | Lluvia observada, por día (mm) |

El caudal del NWM y la PPF (QPF) del NWS provienen de APIs de NOAA sin autenticación ([NWPS](https://api.water.noaa.gov/nwps/v1/docs/), [api.weather.gov](https://www.weather.gov/documentation/services-web-api)); la precipitación observada de MRMS se extrae por punto del servicio IEMRE de la [Red de Mesonet Ambiental de Iowa](https://mesonet.agron.iastate.edu/).

## Agregar una estación

Agrega una fila a `flood_warning/sites.py` con `id`, `name`, `lat`, `lon`, `drainage_sqmi`, `kind`. Luego descarga los datos, entrena y calcula sus umbrales.

## Licencia

[Apache 2.0](./LICENSE).
