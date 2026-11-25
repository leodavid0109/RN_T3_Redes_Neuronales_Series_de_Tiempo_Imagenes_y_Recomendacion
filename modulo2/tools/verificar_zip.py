"""Comprueba si los archivos ilegibles del inventario ya venían dañados en el .zip
publicado o si se dañaron al descomprimir.

Uso (desde modulo2/):
  py tools/verificar_zip.py "C:/Users/User/Downloads/Multi-Class Driver Behavior Image Dataset.zip"

Para cada archivo de data/processed/archivos_ilegibles.csv lee la entrada
directamente del .zip (sin descomprimir a disco), valida su CRC-32 y dice si
su contenido son solo ceros. Escribe data/processed/verificacion_zip.csv.
"""
import sys
import zipfile
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parents[1]


def main(ruta_zip: str) -> None:
    il = pd.read_csv(AQUI / "data/processed/archivos_ilegibles.csv")
    filas = []
    with zipfile.ZipFile(ruta_zip) as z:
        nombres = {n.replace("\\", "/"): n for n in z.namelist()}
        for ruta in il["ruta"]:
            nombre = nombres.get(ruta)
            if nombre is None:
                filas.append(dict(ruta=ruta, en_zip=False))
                continue
            info = z.getinfo(nombre)
            try:
                datos = z.read(nombre)  # zipfile valida el CRC-32 al leer
                crc_ok = True
            except zipfile.BadZipFile:
                datos, crc_ok = b"", False
            filas.append(
                dict(
                    ruta=ruta,
                    en_zip=True,
                    bytes_en_zip=info.file_size,
                    crc_ok=crc_ok,
                    solo_ceros_en_zip=bool(datos) and datos.count(0) == len(datos),
                    cabecera_en_zip=datos[:8].hex(),
                )
            )
    res = pd.DataFrame(filas)
    res.to_csv(AQUI / "data/processed/verificacion_zip.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
