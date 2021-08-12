import json
import csv

header = ['fr_no', 'x', 'y', 'w', 'h', 'cx', 'cy', 'cen_dep', 'dep_avg', 'orient', 'group']

f = open('yolo_db_orientation_20210810_cd_1.json')
data = json.load(f)
f.close()

print(data)

with open('yolo_db_orientation_20210810_cd_1.csv', 'w') as f:
    csv_file = csv.writer(f, delimiter=';')
    csv_file.writerow(header)

    for item in data:
        print(item)
        print(data[item])
        for box in data[item]:
            row_data = [int(item)]
            print(box)
            for idx, value in enumerate(box):
                if idx in [6, 7, 8]:
                    val = float(value)
                else:
                    val = int(value)
                row_data.append(val)
            print(row_data)
            csv_file.writerow(row_data)
# for key, val in data:
#
# f.writerow(item)  # ← changed
#
# f.close()
