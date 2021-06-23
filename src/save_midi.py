from music21 import instrument, note, stream, chord, duration
import os


def save_notes_and_durations(generated, filepath):
    midi_stream = stream.Stream()
    for gen_notes, gen_durations in zip(*generated):
        gen_notes = gen_notes.split('.')[:-1]
        # chord
        if len(gen_notes) > 1:
            cur_chord = []
            for cur_note in gen_notes:
                try:
                    new_note = note.Note(int(cur_note))
                except:
                    new_note = note.Note(cur_note)
                new_note.duration = duration.Duration(gen_durations * 4)
                cur_chord.append(new_note)
            midi_stream.append(chord.Chord(cur_chord))
        elif gen_notes != []:
            try:
                new_note = note.Note(int(gen_notes[0]))
            except:
                new_note = note.Note(gen_notes[0])
            new_note.duration = duration.Duration(gen_durations * 4)
            midi_stream.append(new_note)
        else:
            new_note = note.Rest()
            new_note.duration = duration.Duration(gen_durations * 4)
            midi_stream.append(new_note)
            
    midi_stream = midi_stream.chordify()
    midi_stream.write('midi', fp=os.path.join(filepath + '.mid'))
