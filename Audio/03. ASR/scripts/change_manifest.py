"""
# Пример использования
input_file = "path/to/your/manifest.json"  # Замените на путь к вашему файлу
output_file = "path/to/your/new_manifest.json" # Замените на путь, куда сохранить новый файл
wrap_json_in_list(input_file, output_file)
"""

import json
import os
import shutil

def wrap_json_in_list(input_file, output_file):
    backup_file = input_file + ".bak"

    try:
        # Создаем резервную копию
        shutil.copy2(input_file, backup_file)
        print(f"Created backup: {backup_file}")

        with open(input_file, 'r', encoding='utf-8') as f_in:
            data = []
            for line in f_in:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {line.strip()}. Error: {e}")

        with open(output_file, 'w', encoding='utf-8') as f_out:
            json.dump(data, f_out, indent=4, ensure_ascii=False)

        print(f"Successfully wrapped JSON objects and saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        if os.path.exists(backup_file):
            # Восстанавливаем из резервной копии
            shutil.copy2(backup_file, input_file)
            print(f"Restored original file from backup: {backup_file}")
    finally:
        if os.path.exists(backup_file):
            # Удаляем резервную копию, если все прошло успешно
            try:
                os.remove(backup_file)
            except OSError as e:
                print(f"Could not remove backup file: {e}")



# input_file = "./data/librispeech_russian/train/manifest.json"  # Замените на путь к вашему файлу
# output_file = "./data/librispeech_russian/train/manifest.json" # Замените на путь, куда сохранить новый файл
# wrap_json_in_list(input_file, output_file)

input_file = "./data/librispeech_russian/test/manifest.json"  # Замените на путь к вашему файлу
output_file = "./data/librispeech_russian/test/manifest.json" # Замените на путь, куда сохранить новый файл
wrap_json_in_list(input_file, output_file)

input_file = "./data/librispeech_russian/dev/manifest.json"  # Замените на путь к вашему файлу
output_file = "./data/librispeech_russian/dev/manifest.json" # Замените на путь, куда сохранить новый файл
wrap_json_in_list(input_file, output_file)