import matplotlib.pyplot as plt
import yaml
import os

paths = {
        1.1: '18/2024-01-30-09-17-18',
        1.15: '18/2024-01-30-09-15-15',
        1.2: '18/2024-01-30-08-55-27',
        1.25: '18/2024-01-30-09-10-01',
        1.3: '18/2024-01-30-09-12-55',
        }

paths = { 
        1.1: '18/2024-01-22-12-34-41',
        1.15 : '18/2024-01-22-12-51-43',
        1.2 : '18/2024-01-22-12-44-16',
        1.25 : '18/2024-01-22-12-58-30',
        1.3 : '18/2024-01-22-13-04-18'
        }

pixels = [[4, 3], [15, 0], [9, 12], [7, 15], [2, 7]]
data = {f'Row {r} Col {c}':[] for r, c in pixels}
for p in paths:
    filename = 'results/' + paths[p] + '/thresholds.yaml'
    with open(filename, 'r') as f:
        thresh = yaml.load(f, Loader=yaml.FullLoader)
    print(thresh)
    for r, c in pixels:
        data[f'Row {r} Col {c}'].append(thresh[r][c])
fig = plt.figure(figsize = (9,7))
x = [p for p in paths]
for p in data:
    plt.plot(x, data[p], 'o-', label = p)
plt.legend()
plt.title('Module 18, Manual Threshold Scan')
plt.savefig('outputs/18/multipix_auto_res.png', bbox_inches = 'tight')
