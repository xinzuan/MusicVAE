# Copyright 2023 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import os
import csv

import shutil
from note_seq import midi_io

import tensorflow.compat.v1 as tf
# These preprocess functions are taken from https://github.com/magenta/magenta/blob/main/magenta/scripts/convert_dir_to_note_sequences.py to generate tfrecords from midi file
# This file is slightly adjusted by removing unused function such as convert XML,etc
# In addition, new function to split the groove dataset is implemented


def generate_note_sequence_id(filename, collection_name, source_type):
  """Generates a unique ID for a sequence.

  The format is:'/id/<type>/<collection name>/<hash>'.

  Args:
    filename: The string path to the source file relative to the root of the
        collection.
    collection_name: The collection from which the file comes.
    source_type: The source type as a string (e.g. "midi" or "abc").

  Returns:
    The generated sequence ID as a string.
  """
  filename_fingerprint = hashlib.sha1(filename.encode('utf-8'))
  return '/id/%s/%s/%s' % (
      source_type.lower(), collection_name, filename_fingerprint.hexdigest())

def convert_midi(root_dir, sub_dir, full_file_path):
  """Converts a midi file to a sequence proto.

  Args:
    root_dir: A string specifying the root directory for the files being
        converted.
    sub_dir: The directory being converted currently.
    full_file_path: the full path to the file to convert.

  Returns:
    Either a NoteSequence proto or None if the file could not be converted.
  """
  try:
    sequence = midi_io.midi_to_sequence_proto(
        tf.gfile.GFile(full_file_path, 'rb').read())
  except midi_io.MIDIConversionError as e:
    tf.logging.warning(
        'Could not parse MIDI file %s. It will be skipped. Error was: %s',
        full_file_path, e)
    return None
  sequence.collection_name = os.path.basename(root_dir)
  sequence.filename = os.path.join(sub_dir, os.path.basename(full_file_path))
  sequence.id = generate_note_sequence_id(
      sequence.filename, sequence.collection_name, 'midi')
  tf.logging.info('Converted MIDI file %s.', full_file_path)
  return sequence

def convert_files(root_dir, sub_dir, writer, recursive=False):
  """Converts files.

  Args:
    root_dir: A string specifying a root directory.
    sub_dir: A string specifying a path to a directory under `root_dir` in which
        to convert contents.
    writer: A TFRecord writer
    recursive: A boolean specifying whether or not recursively convert files
        contained in subdirectories of the specified directory.

  Returns:
    A map from the resulting Futures to the file paths being converted.
  """
  dir_to_convert = os.path.join(root_dir, sub_dir)
  tf.logging.info("Converting files in '%s'.", dir_to_convert)
  files_in_dir = tf.gfile.ListDirectory(os.path.join(dir_to_convert))
  recurse_sub_dirs = []
  written_count = 0
  for file_in_dir in files_in_dir:
    tf.logging.log_every_n(tf.logging.INFO, '%d files converted.',
                           1000, written_count)
    full_file_path = os.path.join(dir_to_convert, file_in_dir)
    if (full_file_path.lower().endswith('.mid') or
        full_file_path.lower().endswith('.midi')):
    
      try:
        sequence = convert_midi(root_dir, sub_dir, full_file_path)
       
      except Exception as exc:  # pylint: disable=broad-except
        tf.logging.fatal('%r generated an exception: %s', full_file_path, exc)
        continue
      if sequence:
        writer.write(sequence.SerializeToString())
    else:
      if recursive and tf.gfile.IsDirectory(full_file_path):
        recurse_sub_dirs.append(os.path.join(sub_dir, file_in_dir))
      else:
        tf.logging.warning(
            'Unable to find a converter for file %s', full_file_path)

  for recurse_sub_dir in recurse_sub_dirs:
    convert_files(root_dir, recurse_sub_dir, writer, recursive)

def convert_directory(root_dir, output_file, recursive=False):
  """Converts files to NoteSequences and writes to `output_file`.

  Input files found in `root_dir` are converted to NoteSequence protos with the
  basename of `root_dir` as the collection_name, and the relative path to the
  file from `root_dir` as the filename. If `recursive` is true, recursively
  converts any subdirectories of the specified directory.

  Args:
    root_dir: A string specifying a root directory.
    output_file: Path to TFRecord file to write results to.
    recursive: A boolean specifying whether or not recursively convert files
        contained in subdirectories of the specified directory.
  """
  with tf.io.TFRecordWriter(output_file) as writer:
    convert_files(root_dir, '', writer, recursive)


def split_groove(dir_config):

  # Define the source directory containing the files
  source_directory = dir_config['source_dir']

  # Define the destination directories
  train_directory = dir_config['train_dir']
  test_directory = dir_config['test_dir']
  validation_directory = dir_config['val_dir']

  # Read the CSV file
  csv_file_path = dir_config['csv_file']


  with open(csv_file_path, 'r') as file:
    csv_reader = csv.DictReader(file)

    # Retrieve the header names
    header_names = csv_reader.fieldnames


    # Process each row in the CSV file
    for row in csv_reader:
        # Access the values in each row using the header names
        # Example: access the 'filename' and 'split' values
        filename = row['midi_filename']
        split = row['split']
          
          # Build the source file path
        source_file_path = os.path.join(source_directory, filename)
        
        # Determine the destination directory based on the split value
        if split == 'train':
            destination_directory = train_directory
        elif split == 'test':
            destination_directory = test_directory
        elif split == 'validation':
            destination_directory = validation_directory
        else:
            # Handle unknown split values or errors
            print(f"Unknown split value '{split}' for file '{filename}'")
            continue


        # Build the destination file path
        directory_path, filename = os.path.split(filename)
        destination_file_path = os.path.join(destination_directory, filename)

 
        if not os.path.exists(destination_directory):
          os.makedirs(destination_directory)        
        # Move the file to the appropriate directory
        shutil.move(source_file_path, destination_file_path)
if __name__ == '__main__':

    dir_config={
      'source_dir':'./groove-v1.0.0-midionly/groove',
      'train_dir':'./groove-v1.0.0-midionly/groove/train',
      'test_dir':'./groove-v1.0.0-midionly/groove/test',
      'val_dir':'./groove-v1.0.0-midionly/groove/val',
      'csv_file':'./groove-v1.0.0-midionly/groove/info.csv'

    }

    split_groove(dir_config)

    convert_directory(dir_config['train_dir'],'./groove-v1.0.0-midionly/train_preprocess.tfrecord',recursive=True)
    convert_directory(dir_config['test_dir'],'./groove-v1.0.0-midionly/test_preprocess.tfrecord',recursive=True)
    convert_directory(dir_config['val_dir'],'./groove-v1.0.0-midionly/val_preprocess.tfrecord',recursive=True)
