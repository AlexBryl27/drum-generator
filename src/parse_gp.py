import guitarpro

def get_drum_notes(tab):
    for track in tab.tracks:
        if track.isPercussionTrack:
            break
    
    notes = ['del']
    i = 1
    for measure in track.measures:
        for beat in measure.voices[0].beats:
            # pass empty measures
            if len(measure.voices[0].beats) == 1 and len(beat.notes) == 0:
                if notes[i-1] != 'del':
                    notes.append('del')
                continue
            poly_note = ''
            for note in beat.notes:
                # some cleaning
                if note.value == 40:
                    note.value = 38
                elif note.value == 35:
                    note.value = 36
                elif note.value == 57:
                    note.value = 49
                elif note.value == 59:
                    note.value = 51
                elif note.value > 55:
                    continue
                poly_note += str(note.value) + '.'
            notes.append(poly_note)
            i += 1
    return notes


def drop_rests_from_drum_track(notes):
    fixed_notes = []
    durations = []
    i = -1
    for note in notes:
        if note == 'del':
            fixed_notes.append('del')
            durations.append(1)
            i += 1
        else:
            duration = 1/16
            if note == '':
                durations[i] += 1/16
            else:
                fixed_notes.append(note)
                durations.append(duration)
                i += 1
    return fixed_notes, durations