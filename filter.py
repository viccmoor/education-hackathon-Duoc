from pathlib import Path
from threading import Thread

valid_columns = ["AGNO","MRUN","NOM_RBD","GEN_ALU","EDAD_ALU","PROM_GRAL","ASISTENCIA","SIT_FIN"]

def filter_file(file: Path):
    cf_path = Path("out", f'{file.name}')

    with cf_path.open("w", encoding = "UTF-8") as clean_file:
        with file.open("r", encoding = "UTF-8-sig") as dirty_file:
            ### Headers de los CSVs
            column_indexes = []
            for index, label in enumerate(dirty_file.readline().strip().split(";")):
                if label in valid_columns:
                    column_indexes.append((index, label.strip()))

            print(",".join(valid_columns), file = clean_file)

            for line in dirty_file:
                data = line.strip().split(";")
                valid_data = []

                try:
                    for index, label in column_indexes:
                        valid_data.append(data[index])

                    print(",".join(valid_data), file = clean_file)
                except (ValueError, IndexError):
                    continue
                except KeyboardInterrupt:
                    break

            dirty_file.close()
        clean_file.close()

for dirty_csv_file in Path("in").iterdir():
    file_path = Path("out", f'{dirty_csv_file.name}')
    print(f"Filtering file {str(dirty_csv_file.name)}")

    Thread(
        name = dirty_csv_file.stem,
        target = filter_file,
        args = [dirty_csv_file]
    ).start()
