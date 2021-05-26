import guitarpro

def get_drum_notes(tab, hard_cleaning=False):
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
            prev = str()
            for note in beat.notes:
                # some cleaning
                if hard_cleaning:
                    if note.value in [40, 38, 33, 34]:
                        note.value = 38
                    elif note.value == 35:
                        note.value = 36
                    elif note.value in [41, 43, 45, 47]:
                        note.value = 43
                    elif note.value in [49, 57, 42, 44, 46, 51, 53, 59, 54, 55]:
                        note.value = 46
                    # elif note.value == 59:
                    #     note.value = 51
                    elif note.value > 55:
                        continue
                    elif str(note.value) == prev:
                        prev = str(note.value)
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


def get_notes_and_durations(tab, hard_cleaning=False):
    for track in tab.tracks:
        if track.isPercussionTrack:
            break
    
    notes = ['del']
    durations = [1]
    i = 1
    for measure in track.measures:
        for beat in measure.voices[0].beats:
            # pass empty measures
            if len(measure.voices[0].beats) == 1 and len(beat.notes) == 0:
                if notes[i-1] != 'del':
                    notes.append('del')
                    durations.append(1)
                continue
            poly_note = ''
            prev = str()
            for note in beat.notes:
                # some cleaning
                if hard_cleaning:
                    if note.value in [40, 38, 33, 34]:
                        note.value = 38
                    elif note.value == 35:
                        note.value = 36
                    elif note.value in [41, 43, 45, 47]:
                        note.value = 43
                    elif note.value in [49, 57, 42, 44, 46, 51, 53, 59, 54, 55]:
                        note.value = 46
                    # elif note.value == 59:
                    #     note.value = 51
                    elif note.value > 55:
                        continue
                    elif str(note.value) == prev:
                        prev = str(note.value)
                        continue
                poly_note += str(note.value) + '.'
            durations.append(1 / beat.duration.value)
            notes.append(poly_note)
            i += 1
    return notes, durations