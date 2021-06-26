import os
from collections import Counter
from tqdm.auto import tqdm
import guitarpro
from parse_gp import get_notes_and_durations
import pandas as pd


def parse_files_to_sequences(folderpath, track_name, hard_cleaning=False):
    notes, durations = [], []
    for folder in os.listdir(folderpath):
        for filename in tqdm(os.listdir(folderpath + folder + '/')):
            tab = guitarpro.parse(folderpath + folder + '/' + filename)
            tab_notes, tab_durations = get_notes_and_durations(tab, track_name, hard_cleaning)
            notes += tab_notes
            durations += tab_durations
    return notes, durations


def limit_uniq_notes(notes, durations, limit):
    note_counts = Counter(notes)
    df = pd.DataFrame(note_counts.items(), columns=['note', 'cnt']).sort_values('cnt', ascending=False)
    note_to_use = df.iloc[:limit].note.values
    notes_corr, durations_corr = [], []
    for note, dur in zip(notes, durations):
        if note in note_to_use:
            notes_corr.append(note)
            durations_corr.append(dur)
    return notes_corr, durations_corr


def get_dictionaries(notes, durations):
    note_dictionary = {note: i for i, note in enumerate(set(notes))}
    inv_note_dictionary = {i: note for note, i in note_dictionary.items()}
    notes_to_int = [note_dictionary[note] for note in notes]
    duration_dictionary = {dur: i for i, dur in enumerate(set(durations))}
    inv_dur_dictionary = {i: dur for dur, i in duration_dictionary.items()}
    durations_to_int = [duration_dictionary[dur] for dur in durations]
    return note_dictionary, inv_note_dictionary, notes_to_int, \
        duration_dictionary, inv_dur_dictionary, durations_to_int