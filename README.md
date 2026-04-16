# AUV TAN Path Planning (Re-implementation)

## Cập nhật mới
Đã bổ sung loader DEM thật hỗ trợ:
- `.tif` / `.tiff` (GeoTIFF)
- `.hgt` (SRTM)

File: `dem_loader.py`.

## Cách chạy DEM loader
```bash
python -m pip install -r requirements.txt
python dem_loader.py /path/to/dem_file.tiff
python dem_loader.py /path/to/N21E105.hgt
```

Script sẽ in metadata JSON gồm: shape, CRS, nodata, bounds, resolution, min/max elevation.

## Chạy case studies với DEM thật
```bash
python run_case_studies.py \
  --dem-path /path/to/dem_file.tiff \
  --terrain-size 500 \
  --output-dir outputs_dem
```

Hoặc với HGT:
```bash
python run_case_studies.py \
  --dem-path /path/to/N21E105.hgt \
  --terrain-size 500 \
  --output-dir outputs_hgt
```

### Ghi chú
- `run_case_studies.py` sẽ:
  1. Load DEM thật từ `--dem-path`.
  2. Làm sạch nodata/NaN.
  3. Resample về kích thước vuông `--terrain-size` để tương thích planner hiện tại.
  4. Chạy 4 case studies và xuất plot + JSON report.
- Nếu không truyền `--dem-path`, script fallback về synthetic terrain như trước.

## File chính
- `dem_loader.py`: load DEM thật từ GeoTIFF/HGT.
- `run_case_studies.py`: chạy case-study trên DEM thật hoặc synthetic.
- `requirements.txt`: dependency môi trường.
