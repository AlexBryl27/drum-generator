import json

def drop_rests_from_drum_track(notes):
    """
    It can only be useful in my notation.
    A standard 4/4 bit can be: |36 40 36 40| with quarter durations,
    but I have the same bit: |36 R R R 40 R R R 36 R R R 40 R R R|, where R - are rests and
    all durations 1/16
    """
    fixed_notes = []
    durations = []
    i = -1
    for note in notes:
        duration = 1/16
        if note == '':
            durations[i] += 1/16
        else:
            fixed_notes.append(note)
            durations.append(duration)
            i += 1
    return fixed_notes, durations


def clear_drums(note):
    if note.value in [28, 40, 38, 33, 34, 37, 39]:
        return 38
    elif note.value in [35, 36]:
        return 36
    elif note.value in [41, 43, 45, 47, 48, 50, 65, 66]:
        return 43
    elif note.value in [49, 57, 42, 44, 46, 51, 52, 53, 59, 54, 55]:
        return 46
    else:
        return ''


def note_handler(beat, string_map, track_name='guitar', hard_cleaning=False):
    poly_note = ''
    prev = ''
    note_type = 'normal'
    for note in beat.notes:
        # only for drums
        if track_name[:4] == 'drum':
            if hard_cleaning:
                note.value = str(clear_drums(note))
            else:
                note.value = str(note.value)
            if note.value == prev:
                continue
            prev = note.value
        else:
            note.value = string_map[str(note.string)][note.value]
        poly_note += str(note.value) + '.'
        note_type = note.type.name
    return poly_note, note_type


def get_notes_and_durations(tab, track_name='guitar', hard_cleaning=False):
    
    notes = []
    durations = []
    
    with open('string_map.json', 'r') as f:
        string_map = json.load(f)
    
    for track in tab.tracks:
        if track.name[:3] == track_name[:3]:
            break
    
    if track.name[:3] != track_name[:3]:
        return notes, durations

    for measure in track.measures:
        for beat in measure.voices[0].beats:
            # pass empty measures
            if len(measure.voices[0].beats) == 1 and len(beat.notes) == 0:
                continue
            poly_note, note_type = note_handler(beat, string_map, track_name, hard_cleaning)
            if note_type == 'tie':
                durations[-1] += 1 / beat.duration.value
            else:
                dur_value = beat.duration.value
                dot_dur = 0
                if beat.duration.isDotted:
                    dot_dur = 1 / (dur_value * 2)
                # tuplets is igored
                durations.append((1 / dur_value) + dot_dur)
                notes.append(poly_note)
    return notes, durations