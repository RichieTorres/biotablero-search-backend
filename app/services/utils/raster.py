import base64
import io

from PIL import Image
from rasterstats import zonal_stats
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import rasterize
from pyproj import Transformer
from rio_tiler.io.rasterio import Reader
# from rio_tiler.io import rasterio
import numpy as np
import geopandas as gpd
from typing import Any, Dict, List, Tuple
import rioxarray
from shapely import box
from shapely.geometry import shape, Polygon
import xarray

from geojson_pydantic import geometries
from app.middleware.log_middleware import logger
from app.utils.errors import NotFoundError, ServerError


def crop_raster(
    raster_path: str,
    polygon,
    category: int,
    values: List[int],
    colors: List[str],
) -> str:
    Image.MAX_IMAGE_PIXELS = None

    colormap: Dict[int, Tuple[int, int, int, int]] = {
        value: hex_to_rgba(color) for value, color in zip(values, colors)
    }

    if category not in colormap:
        raise NotFoundError(
            usr_msg="Selected category is not available in values.",
            log_msg=f"Category {category} not found in values.",
        )

    try:
        with Reader(input=raster_path, options={}) as image:
            img = image.feature(polygon)

            color = colormap[category]

            rendered_img = img.render(
                add_mask=True, colormap={category: color}
            )

            if not rendered_img:
                raise NotFoundError(
                    usr_msg="No data available for the selected category.",
                    log_msg=f"No data generated for category {category}.",
                )

            pil_image = Image.open(io.BytesIO(rendered_img))
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format="PNG")
            img_buffer.seek(0)

            img_base64 = base64.b64encode(img_buffer.getvalue()).decode(
                "utf-8"
            )

    except Exception as e:
        logger.error(
            f"Unexpected error rendering category {category}: {str(e)}"
        )
        raise ServerError(
            code=500,
            usr_msg=f"There was an error processing category {category}.",
            e=e,
        )

    return img_base64


def get_raster_values2(cog_path: str, polygon: geometries.MultiPolygon, categories: Dict[str, int]
):

    polygon_geom = shape(polygon)
    with rasterio.open(cog_path) as src:
        # Obtener bounds del polígono
        minx, miny, maxx, maxy = polygon_geom.bounds

        # Crear ventana que cubre solo esa área
        window = from_bounds(minx, miny, maxx, maxy, src.transform)

        # Leer SOLO esa ventana (no todo el archivo)
        data = src.read(1, window=window)

        clean_data = np.where(np.isnan(data), 0, data)

        window_transform = src.window_transform(window)

        # Crear máscara del polígono para esta ventana
        # mask = self._create_polygon_mask(data.shape, polygon_geom, window_transform)
        mask = rasterize(
            [polygon_geom],  # Lista de geometrías
            out_shape=clean_data.shape,  # Forma del array (height, width)
            transform=window_transform,  # Transformación geoespacial
            fill=0,  # Valor fuera del polígono
            default_value=1,  # Valor dentro del polígono
            dtype=np.uint8
        )

        # Aplicar máscara y calcular estadísticas
        masked_data = np.where(mask, clean_data, 0)

        # return self._calculate_coverage_stats(masked_data, window_transform)
        # pixel_area = abs(window_transform[0] * window_transform[4])  # ancho × alto del pixel

        # # Si estás en grados, convertir a metros aproximadamente
        # if abs(window_transform[0]) < 1:  # Probablemente grados
        #     # 1 grado ≈ 111,320 metros en el ecuador
        #     pixel_area_m2 = pixel_area * (111320 ** 2)
        # else:
        #     pixel_area_m2 = pixel_area  # Ya está en metros

        # valid_data = masked_data[masked_data > 0]

        # # Contar pixeles por tipo de cobertura
        # unique_values, counts = np.unique(valid_data, return_counts=True)

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:9377", always_xy=True)

        # Obtener coordenadas de una celda en grados
        pixel_width_degrees = abs(window_transform[0])
        pixel_height_degrees = abs(window_transform[4])

        # Tomar un punto de referencia (centro de los datos)
        center_x = window_transform[2]  # x del centro
        center_y = window_transform[5]  # y del centro

        # Crear las 4 esquinas de un pixel en grados
        corners_geo = [
            (center_x, center_y),
            (center_x + pixel_width_degrees, center_y),
            (center_x + pixel_width_degrees, center_y + pixel_height_degrees),
            (center_x, center_y + pixel_height_degrees)
        ]

        # Reproyectar las esquinas a EPSG:9377
        corners_projected = [transformer.transform(x, y) for x, y in corners_geo]

        pixel_polygon = Polygon(corners_projected)
        pixel_area_m2 = pixel_polygon.area

        print(f"2. Área del pixel reproyectada: {pixel_area_m2} m²")

        # Resto del cálculo igual
        valid_data = masked_data[masked_data > 0]

        if len(valid_data) == 0:
            return {}

        unique_values, counts = np.unique(valid_data, return_counts=True)

        # Calcular áreas
        results = {}
        for value, count in zip(unique_values, counts):
            area_m2 = count * pixel_area_m2
            area_ha = area_m2 / 10000  # Convertir a hectáreas
            print(f"6. Procesando valor {value}...")
            if value in categories.values():
                category_key = [
                    class_name
                    for class_name, val in categories.items()
                    if val == value
                ][0]
                results[category_key] = area_ha

        return results


def get_raster_values(
    raster_path: str,
    polygon: geometries.MultiPolygon,
    categories: Dict[str, int],
) -> Dict[str, Any]:

    gdf = gpd.GeoDataFrame({"geometry": [polygon]}, crs="EPSG:4326")

    target_crs = "EPSG:9377"

    raster = rioxarray.open_rasterio(raster_path, masked=True)

    if isinstance(raster, xarray.DataArray):
        raster_box = box(*raster.rio.bounds())

        if not gdf.geometry.intersects(raster_box).any():
            return {}

    clipped_raster = raster.rio.clip(gdf.geometry, from_disk=True)  # type: ignore -> for Pyright it's an error because open_rasterio can return a list[Dataset] but the list doesn't have the clip function -> https://github.com/corteva/rioxarray/blob/6334ca0584b9ccedaba6026c6dc13bea1d63fb9e/rioxarray/raster_dataset.py#L326

    if clipped_raster.rio.crs != target_crs:
        clipped_raster = clipped_raster.rio.reproject(target_crs)

    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    stats = zonal_stats(
        gdf,
        clipped_raster.values[0],
        affine=clipped_raster.rio.transform(),
        categorical=True,
        nodata=np.nan,
    )

    areas_by_category = stats[0]

    pixel_area_m2 = abs(clipped_raster.rio.transform()[0]) ** 2
    pixel_area_ha = pixel_area_m2 / 10000

    output_data = {}
    for category, pixel_count in areas_by_category.items():
        area_ha = pixel_count * pixel_area_ha
        if category in categories.values():
            category_key = [
                class_name
                for class_name, val in categories.items()
                if val == category
            ][0]
            output_data[category_key] = area_ha

    return output_data


def hex_to_rgba(hex_color: str) -> Tuple[int, int, int, int]:
    if hex_color.startswith("#"):
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 6:
            r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
            return r, g, b, 255
        else:
            raise ValueError(f"Invalid hex color format: {hex_color}")
    raise ValueError(f"Hex color must start with '#': {hex_color}")
