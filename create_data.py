'''
Script to generate the data (train/dev/test splits)
'''

import sys
import os
import argparse

from data import read_config

parser = argparse.ArgumentParser()
parser.add_argument(
  "--config",
  help="path to json config",
  required=True
)
args = parser.parse_args()
config_file_path = args.config
config = read_config(config_file_path)

import random
random.seed(config['data']['seed'])

def check_context_size(slideshow, cur_idx, title_checker, content_checker, config):
  '''
  checks if the current slide has expected number of context slides
  '''
  if config['data']['strict_context_size'] == False:
    return True
  for i in xrange(config['data']['context_size']):
    if config['data']['use_left_context'] == True:
      left_idx = cur_idx-i-1
      if left_idx < 0:
        # left slide absent
        return False
      if config['data']['strict_context_title'] == True and title_checker[left_idx] == False:
        # left slide has not met expected title size
        return False
      if config['data']['strict_context_content'] == True and content_checker[left_idx] == False:
        # left slide has not met expected content size
        return False    
    if config['data']['use_right_context'] == True:
      right_idx = cur_idx+i+1
      if right_idx >= len(slideshow):
        # right slide absent
        return False
      if config['data']['strict_context_title'] == True and title_checker[right_idx] == False:
        # right slide has not met expected title size
        return False
      if config['data']['strict_context_content'] == True and content_checker[right_idx] == False:
        # right slide has not met expected content size
        return False
  return True

def get_context(slideshow, cur_idx, config):
  '''
  get the context slides if any
  '''
  left_context_slides, right_context_slides = [], []
  for i in xrange(config['data']['context_size']):
    if config['data']['use_left_context'] == True:
      left_idx = cur_idx-config['data']['context_size']+i
      if left_idx>=0:
        left_context_slides.append([ left_idx, slideshow[left_idx] ])   
    if config['data']['use_right_context'] == True:
      right_idx = cur_idx+i+1
      if right_idx < len(slideshow):
        right_context_slides.append([ right_idx, slideshow[right_idx] ])
  contexts = {}
  if len(left_context_slides) > 0:
    contexts['left'] = left_context_slides
  if len(right_context_slides) > 0:
    contexts['right'] = right_context_slides
  return contexts

def get_valid_slides(slideshow, config):
  '''
  extract valid slides (along with context) from slideshow
  '''
  slides = []

  # create a checker array corresponding to slide validity
  title_checker, content_checker = [], []
  for i in xrange(len(slideshow)):
    slide_title = slideshow[i].split('$$$')[0]
    validity = True
    if len(slide_title.strip())==0 or len(slide_title.split()) > config['data']['max_trg_length']:
      validity = False
    title_checker.append(validity)
    validity = True
    slide_content = slideshow[i].split('$$$')[1]
    num_words = len(slide_content.split())
    if num_words < config['data']['min_src_content_length'] or num_words > config['data']['max_src_content_length']:
      validity = False
    content_checker.append(validity)

  for i in xrange(len(slideshow)):
    if title_checker[i] == True and content_checker[i] == True:
      slide_title = slideshow[i].split('$$$')[0]
      slide_content = slideshow[i].split('$$$')[1]
      if check_context_size(slideshow, i, title_checker, content_checker, config) == True:
        contexts = get_context(slideshow, i, config)
        slides.append([ i, slideshow[i], contexts ])

  return slides

def get_records(config):
  '''
  create train, val and test slides corpus
  '''
  print 'creating train, val and test splits...'
  categories = config['data']['categories'].split(',')
  cat_2_slides = {}
  reader = open(config['data']['master_file'], 'r')
  for line in reader:
    content = line.strip().split('\t')
    cur_category = content[1]
    if cur_category in categories:
      cur_slideshow_id = content[0]
      num_slides = int(content[2])
      slideshow = []
      for i in xrange(num_slides):
        slideshow.append(content[i+4])
      valid_slides = get_valid_slides(slideshow, config)
      if cur_category not in cat_2_slides:
        cat_2_slides[cur_category] = []
      for slide in valid_slides:
        cat_2_slides[cur_category].append([cur_slideshow_id, slide])
  reader.close()
  train_data, val_data, test_data = [], [], []
  for cur_topic in cat_2_slides:
    cur_records = cat_2_slides[cur_topic]
    random.shuffle(cur_records)
    print(str(len(cur_records))+' records from '+cur_topic)
    num_records = len(cur_records)
    num_train_records = int(config['data']['train_percent'] * num_records)
    num_val_records = int(config['data']['val_percent'] * num_records)
    num_test_records = num_records - num_train_records - num_val_records
    cur_train_data = cur_records[:num_train_records]
    cur_val_data = cur_records[num_train_records:num_train_records+num_val_records]
    cur_test_data= cur_records[num_train_records+num_val_records:num_train_records+num_val_records+num_test_records]
    for item in cur_train_data:
      train_data.append(item)
    for item in cur_val_data:
      val_data.append(item)
    for item in cur_test_data:
      test_data.append(item)    
  print('dataset size = ('+str(len(train_data))+","+str(len(val_data))+","+str(len(test_data))+")")
  return train_data, val_data, test_data

def write_to_disk(records, file, config):
  '''
  write the records to disk
  '''
  w = open(file, 'w')
  for record in records:
    slideshow_id, targ_slide_info = record[0], record[1]
    targ_slide_id, targ_slide_text, targ_slide_context = targ_slide_info[0], targ_slide_info[1], targ_slide_info[2]
    out = str(slideshow_id) + "\t" + str(targ_slide_id) + "$$$" + targ_slide_text
    if 'left' in targ_slide_context:
      left_context = targ_slide_context['left']
      out = out + "\tleft$$$" + str(len(left_context))
      for context_slide in left_context:
        slide_id, slide_content = str(context_slide[0]), context_slide[1]
        out = out + "\t" + slide_id + "$$$" + slide_content
    if 'right' in targ_slide_context:
      right_context = targ_slide_context['right']
      out = out + "\tright$$$" + str(len(right_context))
      for context_slide in right_context:
        slide_id, slide_content = str(context_slide[0]), context_slide[1]
        out = out + "\t" + slide_id + "$$$" + slide_content
    w.write(out.strip()+"\n")
  w.close()

train_data, val_data, test_data = get_records(config)
write_to_disk(train_data, config['data']['train_file'], config)
write_to_disk(val_data, config['data']['dev_file'], config)
write_to_disk(test_data, config['data']['test_file'], config)
