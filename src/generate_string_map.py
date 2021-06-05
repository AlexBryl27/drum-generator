import json


octave = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
inv_octave = octave[::-1]

string_map = {
    7: [inv_octave[i] + '2' for i in range(0, 0-12, -1)] + [inv_octave[i] + '3' for i in range(0, 0-12, -1)] + ['B4'],
    6: [inv_octave[i] + '3' for i in range(7, 7-12, -1)] + [inv_octave[i] + '4' for i in range(7, 7-12, -1)] + ['E5'],
    5: [inv_octave[i] + '3' for i in range(2, 2-12, -1)] + [inv_octave[i] + '4' for i in range(2, 2-12, -1)] + ['A5'],
    4: [inv_octave[i] + '4' for i in range(9, 9-12, -1)] + [inv_octave[i] + '5' for i in range(9, 9-12, -1)] + ['D6'],
    3: [inv_octave[i] + '4' for i in range(4, 4-12, -1)] + [inv_octave[i] + '5' for i in range(4, 4-12, -1)] + ['G6'],
    2: [inv_octave[i] + '4' for i in range(0, 0-12, -1)] + [inv_octave[i] + '5' for i in range(0, 0-12, -1)] + ['B6'],
    1: [inv_octave[i] + '5' for i in range(7, 7-12, -1)] + [inv_octave[i] + '6' for i in range(7, 7-12, -1)] + ['E7']
}

with open('string_map.json', 'w') as f:
    json.dump(string_map, f)