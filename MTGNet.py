# Machine Learning Imports
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import joblib
import itertools
from nltk.util import ngrams
from nltk.stem.porter import *
# Miscellaneous Processing Imports
import re
import os
# Imports for Magic Card Processing
try:
  from sf_price_fetcher import fetcher
except:
  print("Import error: Run the 'pip install sf_price_fetcher' cell above.")
try:
  import mtgsdk as mtg
  from mtgsdk import Card, Set, Type, Supertype, Subtype
except:
  print("Import error: Run the 'pip install mtgsdk' cell above.")

# Initialize Global Variables
NONTEXT_SCALER = preprocessing.MinMaxScaler()
NONTEXT_CONDENSED_SCALER = preprocessing.MinMaxScaler()
TEXT_SCALER = preprocessing.MinMaxScaler()
# PRICE_SCALER = preprocessing.RobustScaler()

################################################################################
# LOADING CARD DATA INTO DATAFRAME
################################################################################

# Convert the input list of card objects to a csv file containing all relevant card statistics and information.
# Input: cards -- list of Card objects (mtgsdk)
#        filename -- the name of the file to save the card csv file to
def write_cards_csv(cards, filename="cards.csv"):
  # Remove any duplicate card printings.
  unique_cards = []
  for card in cards:
    if card.name in [u.name for u in unique_cards]:
      continue
    unique_cards.append(card)
  cards = unique_cards
  # Create a pandas dataframe with the relevant card attributes.
  df = pd.DataFrame(columns=['Name', 'ManaCost', 'Power', 'Toughness', 'Loyalty', 'Text', 'SuperTypes', 'CardTypes', 'SubTypes', 'Set', 'Legalities'],index=range(0, len(cards)))
  for i, card in enumerate(cards):
    df.loc[i, "Name"] =       card.name
    df.loc[i, "ManaCost"] =   card.mana_cost
    df.loc[i, "Power"] =      card.power
    df.loc[i, "Toughness"] =  card.toughness
    df.loc[i, "Loyalty"] =    card.loyalty
    df.loc[i, "Text"] =       card.text
    df.loc[i, "Set"] =        card.set
    df.loc[i, "Legalities"] = card.legalities
    # Preprocess card type.
    this_type = card.type.replace("—","")
    if card.subtypes is not None:
      for subtype in card.subtypes:
        this_type = this_type.replace(subtype, "")
    if card.supertypes is not None:
      for supertype in card.supertypes:
        this_type = this_type.replace(supertype, "")
    this_type = this_type.strip()
    this_type = this_type.replace(" ", "-")
    df.loc[i, "CardTypes"] = this_type
    # Preprocess card supertype.
    this_supertype = card.supertypes
    if this_supertype is None or len(this_supertype)==0:
      this_supertype = "-"
    else:
      combined_type = this_supertype[0]
      for supertype in this_supertype[1:]:
        combined_type+= ("-"+supertype)
      this_supertype = combined_type
    df.loc[i, "SuperTypes"] = this_supertype
    # Preprocess card subtype.
    this_subtype = card.subtypes
    if this_subtype is None or len(this_subtype)==0:
      this_subtype = "-"
    else:
      combined_type = this_subtype[0]
      for subtype in this_subtype[1:]:
        combined_type+= ("-"+subtype)
      this_subtype = combined_type
    df.loc[i, "SubTypes"] = this_subtype
  # Write dataframe to a file named cards.csv, or the input filename
  df.to_csv(filename)

# Creates a file called cards.csv containing card information for all MTG cards.
# Uses Card.all() to get all card information at once -- prone to HTTP errors.
def create_cards_csv():
  server_connect_attempts = 5
  while server_connect_attempts >= 0:
    try:
        cards = Card.all()
        server_connect_attempts = -1
    except:
      server_connect_attempts += -1
  write_cards_csv(cards)

# Creates a csv file for each MTG set in Set.all().
# Gets card information one set at a time -- less prone to HTTP errors than create_cards_csv.
def create_cards_csv_by_set():
  sets = [s.code for s in Set.all()]
  already_created_csvs = [c for c in os.listdir() if c.endswith(".csv")]
  for s in sets:
    filename = ('cards_'+s+'.csv')
    if filename not in already_created_csvs:
      server_connect_attempts = 3
      while server_connect_attempts >= 0:
        try:
          cards = Card.where(set=s).all()
          server_connect_attempts = -1
        except:
          server_connect_attempts += -1
      write_cards_csv(cards, filename)
  combine_set_csvs()

# Combine all of the csv files containing card data for a single set into one csv file containing all card data.
def combine_set_csvs(set_csvs=[], output_filename="cards.csv"):
  if len(set_csvs)==0:
    set_csvs = [c for c in os.listdir() if c.endswith(".csv")]
  merged_df = pd.concat(map(pd.read_csv, [csv for csv in set_csvs]), ignore_index=True)
  merged_df.to_csv(output_filename)

################################################################################
# TEXT PREPROCESSING
################################################################################

# Preprocess the input text by replacing all symbols and names with text descriptions that can be processed by word2vec.
def preprocess_card_text(card_name, card_text):
  if type(card_text) != str:
    return card_text
  card_text = separate_keyword_lists(card_text)
  card_text = clean_card_text(card_text)
  card_text = replace_power_toughness(card_text)
  card_text = replace_card_name(card_name, card_text)
  card_text = replace_tap_symbols(card_text)
  card_text = replace_mana_symbols(card_text)
  card_text = replace_chapter_symbols(card_text)
  card_text = replace_loyalty_symbols(card_text)
  card_text = replace_energy_symbols(card_text)
  card_text = replace_numbers(card_text)
  card_text = replace_reminder_text(card_text)
  card_text = replace_in_context_phrases(card_text)
  return card_text

# Returns a list where each item contains partitions of the input text divided by '\n'
def partition_by_newline(card_text):
  newline_partition = card_text.partition('\n')
  partitioned_text = [newline_partition[0]]
  while len(newline_partition[2])>0:
    newline_partition = newline_partition[2].partition('\n')
    partitioned_text.append(newline_partition[0])
  return partitioned_text

# Separate any comma separated list of keywords into new lines.
def separate_keyword_lists(card_text):
  keywords = ['deathtouch', 'defender', 'double strike', 'first strike', 'flash', 'flying', 'haste', 'hexproof', 'indestructible', 'lifelink', 'menace', 'reach', 'trample', 'vigilance', 'shroud']
  partitions = partition_by_newline(card_text)
  for i, part in enumerate(partitions):
    is_list_of_keywords = True
    for p in [s.lower() for s in part.split(", ")]:
      if p in keywords:
        continue
      if "ward" in p and len(p)<=10:
        continue
      if "protection from " in p and len(p.replace("protection from ", "")) <= 10:
        continue
      is_list_of_keywords = False
      break
    if is_list_of_keywords:
      partitions[i] = part.replace(", ", "\n")
  card_text = "\n".join(partitions)
  return card_text

# Return all possible orders of new lines in the card text.
def newline_permutations(card_text):
  if type(card_text)!=str:
    return [card_text]
  card_text = separate_keyword_lists(card_text)
  partitioned_text = partition_by_newline(card_text)
  permutations = list(itertools.permutations(partitioned_text))
  permuted_text = ['\n'.join(p) for p in permutations]
  return permuted_text

# Miscellaneous basic text cleaning.
def clean_card_text(card_text):
  card_text = card_text.lower() # Lowercase all input text 
  card_text = card_text.replace("his or her", "their") # Replace newline characters with ends of sentences
  card_text = card_text.replace("'s", " 's")
  return card_text

# Removes newlines, replaces them with periods.
def clean_newlines(card_text):
  card_text = card_text.replace(".\n", ". ") # Replace newline characters with ends of sentences
  card_text = card_text.replace("\n", ". ")  # Replace newline characters with ends of sentences
  return card_text

# Look through the card_text and replace any power/toughness pairs with text descriptions of those numbers.
# "1/2" --> "one power two toughness"
# "3/x" --> "three power variable toughness"
# "0/4" --> "zero power four toughness"
# "*/*" --> "star power star toughness"
# "+1/-1" --> "plus one power minus one toughness"
def replace_power_toughness(card_text):
  # Every power and toughness will always be one of these symbols:
  symbol_dict = {'0':'zero','1':'one','2':'two','3':'three','4':'four','5':'five','6':'six','7':'seven','8':'eight','9':'nine','10':'ten','11':'eleven','12':'twelve','13':'thirteen','14':'fourteen','15':'fifteen','16':'sixteen','17':'seventeen','18':'eighteen','19':'nineteen','20':'twenty','21':'twenty one','22':'twenty two','23':'twenty three','24':'twenty four','25':'twenty five','x':'variable','*':'star','+':'plus','-':'minus'}
  # Find every index of a slash in the text:
  slash_indices = [i for i in range(len(card_text)) if card_text.startswith('/', i)]
  slash_indices = [s for s in slash_indices if (s > 0 and s<(len(card_text)-1))]
  # For every slash, make sure that it's a power/toughness by checking that the previous and next characters are in the symbol dictionary above.
  slash_indices = [s for s in slash_indices if card_text[s-1] in symbol_dict.keys() and card_text[s+1] in symbol_dict.keys()]
  # What we have now is a list of indices of power/toughness slashes (e.g., [5, 20]).
  # We want to convert this to a list of power/toughness strings (e.g., ["1/1", "3/2"])
  power_toughness = []
  for s in slash_indices:
    start_index = s-1
    while start_index-1>=0 and card_text[start_index-1] in symbol_dict.keys():
      start_index = start_index-1
    end_index = s+2
    while end_index<len(card_text) and card_text[end_index] in symbol_dict.keys():
      end_index = end_index+1
    power_toughness.append(card_text[start_index:end_index])
  # We now have every index of a power/toughness in the text.
  # For each power and toughness string, turn that into a text description of that power and toughness (e.g, "1/1" --> "one power one toughness")
  power_toughness = sorted(power_toughness, key=len)
  power_toughness.reverse()  # Need to sort for an edge case: longer p/t strings should be replaced first
  for pt in power_toughness: # Each pt looks like this: "2/3"
    power_colon_toughness = pt.partition("/")
    power_desc, toughness_desc = "",""
    if len(power_colon_toughness[0])==1 or ("+" not in power_colon_toughness[0] and "-" not in power_colon_toughness[0]):
      power_desc = symbol_dict[power_colon_toughness[0]]
    else:
      if "+" in power_colon_toughness[0]:
        partition = power_colon_toughness[0].partition("+")
      elif "-" in power_colon_toughness[0]:
        partition = power_colon_toughness[0].partition("-")
      power_desc = symbol_dict[partition[1]] + " " + symbol_dict[partition[2]]
    if len(power_colon_toughness[2])==1 or ("+" not in power_colon_toughness[2] and "-" not in power_colon_toughness[2]):
      toughness_desc = symbol_dict[power_colon_toughness[2]]
    else:
      if "+" in power_colon_toughness[2]:
        partition = power_colon_toughness[2].partition("+")
      elif "-" in power_colon_toughness[2]:
        partition = power_colon_toughness[2].partition("-")
      toughness_desc = symbol_dict[partition[1]] + " " + symbol_dict[partition[2]]
    text_description = power_desc + " power " + toughness_desc + " toughness"
    card_text = card_text.replace(pt, text_description) # Replace every instance of that power/toughness symbol with a text string description
  return card_text

# Look through the card_text and replace any numbers with text descriptions of those numbers. (Probably want to go 0-100)
# (e.g., replace "1" with "one", "40" with "forty")
# NOTE: you probably want to replace larger numbers first. If you start by replacing all single digit numbers, you could mistakenly replace part of a 2 or 3 digit number.
def replace_numbers(card_text):
  dice_dict = {"d20":"die twenty", "d16":"die sixteen", "d12":"die twelve", "d10":"die ten", "d8":"die eight", "d6":"die six", "d4":"die four"}
  for dice in dice_dict.keys():
    card_text = card_text.replace(dice, dice_dict[dice])
  number_dict = {'100':'one hundred','99':'ninety nine','98':'ninety eight','97':'ninety seven','96':'ninety six','94':'ninety four','93':'ninety three','92':'ninety two','91':'ninety one','90':'ninety','89':'eighty nine','88':'eighty eight','87':'eighty seven','86':'eighty six','85':'eighty five','84':'eighty four','83':'eighty three','82':'eighty two','81':'eighty one','80':'eighty','79':'seventy nine','78':'seventy eight','77':'seventy seven','76':'seventy six','75':'seventy five','74':'seventy four','73':'seventy three','72':'seventy two', '71':'seventy one', '70':'seventy', '69':'sixty nine', '68':'sixty eight', '67':'sixty seven', '66':'sixty six', '65':'sixty five', '64':'sixty four', '63':'sixty three', '62':'sixty two','61':'sixty one', '60':'sixty', '59':'fifty nine', '58':'fifty eight', '57':'fifty seven', '56':'fifty six', '55':'fifty five', '54':'fifty four', '53':'fifty three', '52':'fifty two', '51':'fifty one', '50':'fifty', '49':'forty nine', '48':'forty eight', '47':'forty seven', '46':'forty six', '45':'forty five', '44':'forty four', '43':'forty three', '42':'forty two', '41':'forty one', '40':'forty', '39':'thirty nine', '38':'thirty eight', '37':'thirty seven', '36':'thirty six', '35':'thirty five', '34':'thirty four', '33':'thirty three', '32':'thirty two', '31':'thirty one', '30':'thirty', '29':'twenty nine', '28':'twenty eight', '27':'twenty seven', '26':'twenty six', '25':'twenty five', '24':'twenty four', '23':'twenty three', '22':'twenty two', '21':'twenty one', '20':'twenty', '19':'nineteen', '18':'eighteen', '17':'seventeen', '16':'sixteen', '15':'fifteen', '14':'fourteen', '13':'thirteen', '12':'twelve', '11':'eleven', '10':'ten', '9':'nine', '8':'eight', '7':'seven', '6':'six', '5':'five', '4':'four', '3':'three', '2':'two', '1':'one'}
  for integer in number_dict.keys():
    card_text = card_text.replace(integer, number_dict[integer])
  return card_text

# Look through the card_text for any instances of card_name, or part of the card name before a comma. (e.g., If the card is called "Quintorius, Loremage", then look for any instances of "Quintorius, Loremage" or just "Quintorius")
# Replace those instances with the word "self"
# FIRST, replace any instance of the FULL NAME (e.g., "Quintorius, Loremage") with "self"
# NEXT, find where a comma occurs in the name (if at all) -- I recommend you use string partition function
#   "Quintorius, Loremage".partition(",")
#   ["Quintorius", ",", " Loremage"]
#     Take the first part of the partition and replace THAT with "self"
def replace_card_name(card_name, card_text):
  card_text = card_text.replace(card_name, "self")
  non_comma_name = card_name.partition(",")[0]
  card_text = card_text.replace(non_comma_name, "self")
  return card_text

# Look through the card_text for any instances of tap/untap symbols and replace them with text describing those symbols.
# {t} --> "tap self"
# {q} --> "untap self"
def replace_tap_symbols(card_text):
  text ={'{t}':'tap self', '{q}':'untap self'}
  for t in text.keys():
    card_text = card_text.replace(t, text[t])
  return card_text

# Look through the card_text for any instances of a mana symbol and replace them with text describing those symbols.
# {w/u/p}, ... --> "two life or one white or blue mana", ...
def replace_mana_symbols(card_text):
  white_dict = {"{w}{w}{w}{w}{w}":"five white mana ", "{w}{w}{w}{w}":"four white mana ", "{w}{w}{w}":"three white mana ", "{w}{w}":"two white mana ", "{w}":"one white mana "}
  blue_dict =  {"{u}{u}{u}{u}{u}":"five blue mana ",  "{u}{u}{u}{u}":"four blue mana ",  "{u}{u}{u}":"three blue mana ",  "{u}{u}":"two blue mana ",  "{u}":"one blue mana "}
  black_dict = {"{b}{b}{b}{b}{b}":"five black mana ", "{b}{b}{b}{b}":"four black mana ", "{b}{b}{b}":"three black mana ", "{b}{b}":"two black mana ", "{b}":"one black mana "}
  red_dict =   {"{r}{r}{r}{r}{r}":"five red mana ",   "{r}{r}{r}{r}":"four red mana ",   "{r}{r}{r}":"three red mana ",   "{r}{r}":"two red mana ",   "{r}":"one red mana "}
  green_dict = {"{g}{g}{g}{g}{g}":"five green mana ", "{g}{g}{g}{g}":"four green mana ", "{g}{g}{g}":"three green mana ", "{g}{g}":"two green mana ", "{g}":"one green mana "}
  snow_dict =  {"{s}{s}{s}{s}{s}":"five snow mana ",  "{s}{s}{s}{s}":"four snow mana ",  "{s}{s}{s}":"three snow mana ",  "{s}{s}":"two snow mana ",  "{s}":"one snow mana "}
  colorless_dict = {"{c}{c}{c}{c}{c}":"five colorless mana ", "{c}{c}{c}{c}":"four colorless mana ", "{c}{c}{c}":"three colorless mana ", "{c}{c}":"two colorless mana ", "{c}":"one colorless mana "}
  generic_dict = {"{x}":"variable generic mana ", "{0}":"zero generic mana ", "{1}":"one generic mana ", "{2}":"two generic mana ", "{3}":"three generic mana ", "{4}":"four generic mana ", "{5}":"five generic mana ", "{6}":"six generic mana ", "{7}":"seven generic mana ", "{8}":"eight generic mana ", "{9}":"nine generic mana ", "{10}":"ten generic mana ", "{11}":"eleven generic mana ", "{12}":"twelve generic mana ", "{13}":"thirteen generic mana ", "{14}":"fourteen generic mana ", "{15}":"fifteen generic mana ", "{16}":"sixteen generic mana "}
  hybrid_dict = {"{w/u}{w/u}{w/u}":"three white or blue mana ",  "{w/u}{w/u}":"two white or blue mana ",  "{w/u}":"one white or blue mana ",
                 "{u/w}{u/w}{u/w}":"three white or blue mana ",  "{u/w}{u/w}":"two white or blue mana ",  "{u/w}":"one white or blue mana ",
                 "{w/b}{w/b}{w/b}":"three white or black mana ", "{w/b}{w/b}":"two white or black mana ", "{w/b}":"one white or black mana ",
                 "{b/w}{b/w}{b/w}":"three white or black mana ", "{b/w}{b/w}":"two white or black mana ", "{b/w}":"one white or black mana ",
                 "{w/r}{w/r}{w/r}":"three white or red mana ",   "{w/r}{w/r}":"two white or red mana ",   "{w/r}":"one white or red mana ",
                 "{r/w}{r/w}{r/w}":"three white or red mana ",   "{r/w}{r/w}":"two white or red mana ",   "{r/w}":"one white or red mana ",
                 "{w/g}{w/g}{w/g}":"three white or green mana ", "{w/g}{w/g}":"two white or green mana ", "{w/g}":"one white or green mana ",
                 "{g/w}{g/w}{g/w}":"three white or green mana ", "{g/w}{g/w}":"two white or green mana ", "{g/w}":"one white or green mana ",
                 "{u/b}{u/b}{u/b}":"three blue or black mana ",  "{u/b}{u/b}":"two blue or black mana ",  "{u/b}":"one blue or black mana ",
                 "{b/u}{b/u}{b/u}":"three blue or black mana ",  "{b/u}{b/u}":"two blue or black mana ",  "{b/u}":"one blue or black mana ",
                 "{u/r}{u/r}{u/r}":"three blue or red mana ",    "{u/r}{u/r}":"two blue or red mana ",    "{u/r}":"one blue or red mana ",
                 "{r/u}{r/u}{r/u}":"three blue or red mana ",    "{r/u}{r/u}":"two blue or red mana ",    "{r/u}":"one blue or red mana ",
                 "{u/g}{u/g}{u/g}":"three blue or green mana ",  "{u/g}{u/g}":"two blue or green mana ",  "{u/g}":"one blue or green mana ",
                 "{g/u}{g/u}{g/u}":"three blue or green mana ",  "{g/u}{g/u}":"two blue or green mana ",  "{g/u}":"one blue or green mana ",
                 "{b/r}{b/r}{b/r}":"three black or red mana ",   "{b/r}{b/r}":"two black or red mana ",   "{b/r}":"one black or red mana ",
                 "{r/b}{r/b}{r/b}":"three black or red mana ",   "{r/b}{r/b}":"two black or red mana ",   "{r/b}":"one black or red mana ",
                 "{b/g}{b/g}{b/g}":"three black or green mana ", "{b/g}{b/g}":"two black or green mana ", "{b/g}":"one black or green mana ",
                 "{g/b}{g/b}{g/b}":"three black or green mana ", "{g/b}{g/b}":"two black or green mana ", "{g/b}":"one black or green mana ",
                 "{r/g}{r/g}{r/g}":"three red or green mana ",   "{r/g}{r/g}":"two red or green mana ",   "{r/g}":"one red or green mana ",
                 "{g/r}{g/r}{g/r}":"three red or green mana ",   "{g/r}{g/r}":"two red or green mana ",   "{g/r}":"one red or green mana "}
  gen_hy_dict = {"{2/w}":"two generic or one white mana ", "{2/u}":"two generic or one blue mana ", "{2/b}":"two generic or one black mana ", "{2/r}":"two generic or one red mana ", "{2/g}":"two generic or one green mana "}
  phyrex_dict = {"{w/p}{w/p}{w/p}": "three white mana or six life ", "{w/p}{w/p}": "two white mana or four life ", "{w/p}": "one white mana or two life ",
                 "{u/p}{u/p}{u/p}": "three blue mana or six life ",  "{u/p}{u/p}": "two blue mana or four life ",  "{u/p}": "one blue mana or two life ",
                 "{b/p}{b/p}{b/p}": "three black mana or six life ", "{b/p}{b/p}": "two black mana or four life ", "{b/p}": "one black mana or two life ",
                 "{r/p}{r/p}{r/p}": "three red mana or six life ",   "{r/p}{r/p}": "two red mana or four life ",   "{r/p}": "one red mana or two life ",
                 "{g/p}{g/p}{g/p}": "three green mana or six life ", "{g/p}{g/p}": "two green mana or four life ", "{g/p}": "one green mana or two life "}
  phy_hy_dict = {"{w/u/p}": "one white or blue mana or two life ",  "{u/w/p}": "one white or blue mana or two life ",
                 "{w/b/p}": "one white or black mana or two life ", "{b/w/p}": "one white or black mana or two life ",
                 "{w/r/p}": "one white or red mana or two life ",   "{r/w/p}": "one white or red mana or two life ",
                 "{w/g/p}": "one white or green mana or two life ", "{g/w/p}": "one white or green mana or two life ",
                 "{u/b/p}": "one blue or black mana or two life ",  "{b/u/p}": "one blue or black mana or two life ",
                 "{u/r/p}": "one blue or red mana or two life ",    "{r/u/p}": "one blue or red mana or two life ",
                 "{u/g/p}": "one blue or green mana or two life ",  "{g/u/p}": "one blue or green mana or two life ",
                 "{b/r/p}": "one black or red mana or two life ",   "{r/b/p}": "one black or red mana or two life ",
                 "{b/g/p}": "one black or green mana or two life ", "{g/b/p}": "one black or green mana or two life ",
                 "{r/g/p}": "one red or green mana or two life ",   "{g/r/p}": "one red or green mana or two life "}
  mana_dict_list = [white_dict, blue_dict, black_dict, red_dict, green_dict, snow_dict, colorless_dict, generic_dict, hybrid_dict, gen_hy_dict, phyrex_dict, phy_hy_dict]
  for this_dict in mana_dict_list:
    for k in this_dict.keys():
      card_text = card_text.replace(k, this_dict[k])
  return card_text

# Look through the card_text for any instances of Saga chapter symbols and replace them with text describing those symbols.
# "i —" --> "chapter one —"
# "ii —" --> "chapter two —" ...
# Up to five chapters to be safe
# WARNING -- the hyphen used in the text examples I gave above is a special type of hyphen, so make sure you don't use a regular hyphen -
def replace_chapter_symbols(card_text):
  chapter_dict = {"iv —": "chapter four —", "v —": "chapter five —", "iii —": "chapter three —", "ii —": "chapter two —", "i —": "chapter one —"}
  for roman_numerals in chapter_dict.keys():
    card_text = card_text.replace(roman_numerals, chapter_dict[roman_numerals])
  return card_text

# Look through the card_text for any instances of loyalty symbols and replace them with text describing those symbols.
# "[+1]:" --> "plus one loyalty:"
# "[0]:" --> "plus zero loyalty:"
# "[−6]:" --> "minus six loyalty:"
# "[-x]:" --> "minus variable loyalty:"
def replace_loyalty_symbols(card_text):
  loyalty_dict = {"[0]":"plus zero loyalty ",
                  "[+x]":"plus variable loyalty ",  "[+0]":"plus zero loyalty ",  "[+1]":"plus one loyalty ",  "[+2]":"plus two loyalty ",  "[+3]":"plus three loyalty ",  "[+4]":"plus four loyalty ",  "[+5]":"plus five loyalty ",  "[+6]":"plus six loyalty ",  "[+7]":"plus seven loyalty ",
                  "[-x]":"minus variable loyalty ", "[-0]":"minus zero loyalty ", "[-1]":"minus one loyalty ", "[-2]":"minus two loyalty ", "[-3]":"minus three loyalty ", "[-4]":"minus four loyalty ", "[-5]":"minus five loyalty ", "[-6]":"minus six loyalty ", "[-7]":"minus seven loyalty ", "[-8]":"minus eight loyalty ", "[-9]":"minus nine loyalty ", "[-10]":"minus ten loyalty ", "[-11]":"minus eleven loyalty ", "[-12]":"minus twelve loyalty ", "[-13]":"minus thirteen loyalty ", "[-14]":"minus fourteen loyalty ", "[-15]":"minus fifteen loyalty "}
  for loyalty in loyalty_dict.keys():
    card_text = card_text.replace(loyalty, loyalty_dict[loyalty])
  return card_text

# Look through the card_text for any instances of energy symbols and replace them with text describing those symbols.
# {e} --> "energy counter"
# {e}{e} --> "two energy counters"
# {e}{e}{e} --> "three energy counters"
# Go up to five, let's say 7 to be safe
def replace_energy_symbols(card_text):
  energy_dict = {"{e}{e}{e}{e}{e}{e}{e}": "seven energy counters", "{e}{e}{e}{e}{e}{e}": "six energy counters", "{e}{e}{e}{e}{e}": "five energy counters", "{e}{e}{e}{e}": "four energy counters", "{e}{e}{e}": "three energy counters", "{e}{e}": "two energy counters", "{e}": "one energy counter"}
  for e in energy_dict.keys():
    card_text = card_text.replace(e, energy_dict[e])
  return card_text

# TODO -- should we remove all reminder text, or always add reminder text for keywords?
# Look through the card_text for any text block in between parentheses and remove it.
# e.g. "Example text (this is example text)" --> "Example text"
def replace_reminder_text(card_text):
  return card_text

# Replace certain common phrases with single-word descriptions of those phrases.
def replace_in_context_phrases(card_text):
  phrase_dict = {
      ("enter the battlefield", "enters the battlefield", "entered the battlefield", "entering the battlefield"): "materialize",
      ("leave the battlefield", "leaves the battlefield", "left the battlefield", "leaving the battlefield"): "vaporize",
      ("onto the battlefield", "on the battlefield", "to the battlefield"): "hither",
      ("additional cost",): "tariff",
      ("first strike",): "swiftness",
      ("double strike",): "breakneck",
      ("less to cast",): "cheaper",
      ("more to cast",): "pricier",
      ("of any color",): "motley",
      ("in any combination of colors",): "polychrome",
      ("mana pool",): "resources"
      # Others: mana value, converted mana cost, mana cost, under your control, your control, you control, you don't control, an opponent controls, its controller, basic land, +1/+1 counter, -1/-1 counter, end of your turn, end of your next turn, end of turn, combat damage, this turn, next turn, main phase, combat phase, draw step, end step
  }
  for phrase_tup in phrase_dict.keys():
    for phrase in phrase_tup:
      card_text = card_text.replace(phrase, phrase_dict[phrase_tup])
  return card_text

################################################################################
# WORD EMBEDDINGS
################################################################################

# Splits the input text into tokens, where each token is a word or punctuation
def split_text_into_tokens(text):
  return re.findall(r"[\w']+|[,.?!;:/()]", text)

# Build a vocabulary dictionary with all words and punctuation in the input text.
# The dictionary is of the form {word:index}.
def build_vocabulary(text):
  tokens = split_text_into_tokens(text)
  vocabulary_dict = {}
  index = 1
  for token in tokens:
    if token not in vocabulary_dict.keys():
      vocabulary_dict[token] = index
      index += 1
  return vocabulary_dict

# Build an embedding dictionary with all words and punctuation in the input text.
# The dictionary is of the form {word: embedding vector}.
# (e.g., embedding_dict["word"] --> np array of shape (dimension,))
def build_embedding_vocabulary(text, dimension=50):
  vocabulary = build_vocabulary(text)
  embedding_matrix = build_embedding_matrix(vocabulary, dimension)
  embedding_dict = {k:embedding_matrix[v] for (k,v) in zip(vocabulary.keys(), range(1,1+embedding_matrix.shape[0]))}
  embedding_array = np.concatenate([v.reshape(1,v.shape[0]) for v in embedding_dict.values()],axis=0)
  global TEXT_SCALER
  TEXT_SCALER.fit(embedding_array)
  embedding_dict = {k:TEXT_SCALER.transform(v.reshape(1, -1)) for (k,v) in zip(embedding_dict.keys(), embedding_dict.values())}
  embedding_dict = {k:v.reshape((v.shape[1],)) for (k,v) in zip(embedding_dict.keys(), embedding_dict.values())}
  return embedding_dict

# Builds a matrix where the ith row corresponds to the word embedding of word i
# in the input vocabulary.
# The number of columns in the matrix is equal to the dimension of the embedding.
# Modified code from GeeksForGeeks.org: https://www.geeksforgeeks.org/pre-trained-word-embedding-using-glove-in-nlp-models/
def build_embedding_matrix(vocabulary, dimension=50):
    vocab_size = len(vocabulary)+1
    glove_filename = "glove.6B."+str(dimension)+"d.txt"
    try:
      open(glove_filename)
    except:
      try:
        open("drive/MyDrive/MTGNetData/"+glove_filename)
        glove_filename = "drive/MyDrive/MTGNetData/"+glove_filename
      except:
        glove_filename = "glove.6B.50d.txt"
        dimension = 50
        try:
          open(glove_filename)
          print("WARNING: could not find a glove text file with the specified dimension. Using dimension 50.")
        except:
          print("ERROR: glove file of the specified dimension not loaded.")
          return
    embedding_matrix = np.zeros((vocab_size, dimension))
    with open(glove_filename, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in vocabulary:
                embedding_matrix[vocabulary[word]] = np.array(vector, dtype=np.float32)[:dimension]
    return embedding_matrix

# For the input token, returns a word embedding of the specified dimension.
# If the token is in the input vocabulary embeddings_dict, retrieves the token from the dictionary.
# Otherwise, retrieves the token from the glove text file.
def get_embedding_from_token(token, embeddings_dict={}, dimension=None):
  if token in embeddings_dict.keys():
    return embeddings_dict[token]
  if dimension is None:
    if len(embeddings_dict)==0:
      dimension = 50
    else:
      dimension = embeddings_dict[list(embeddings_dict.keys())[0]].shape[0]
  glove_filename = "glove.6B."+str(dimension)+"d.txt"
  try:
    open(glove_filename)
  except:
    try:
      open("drive/MyDrive/MTGNetData/"+glove_filename)
      glove_filename = "drive/MyDrive/MTGNetData/"+glove_filename
    except:
      glove_filename = "glove.6B.50d.txt"
      dimension = 50
      try:
        open(glove_filename)
        print("WARNING: could not find a glove text file with the specified dimension. Using dimension 50.")
      except:
        try:
          open("drive/MyDrive/MTGNetData/"+glove_filename)
          glove_filename = "drive/MyDrive/MTGNetData/"+glove_filename
        except:
          print("ERROR: glove file of the specified dimension not loaded.")
          return np.zeros((dimension,))
  with open(glove_filename, encoding="utf8") as f:
      for line in f:
          word, *vector = line.split()
          if word == token:
              try:
                return TEXT_SCALER.transform(np.array(vector, dtype=np.float32)[:dimension].reshape(1, -1))
              except:
                return np.array(vector, dtype=np.float32)[:dimension]
  return np.zeros((dimension,))

# For the input text (any length), returns an NxM array of word embeddings.
# (N is the number of tokens in the input text. M is the dimension of the embedding.)
def get_embeddings_array_from_text(text, embeddings_dict={}, dimension=None):
  tokens = split_text_into_tokens(text)
  if dimension is None:
    dimension = embeddings_dict[list(embeddings_dict.keys())[0]].shape[0]
  embedding = np.zeros((len(tokens), dimension))
  for (i,token) in enumerate(tokens):
    embedding[i,:] = get_embedding_from_token(token, embeddings_dict, dimension)
  return embedding

# For the input subtypes text, returns an NxM array of word embeddings.
# (N is one more than the number of subtypes. M is the dimension of the embedding.)
# An array of zeros is appended at the end of the subtypes to indicate the "end" of the subtype array. 
# If there are no subtypes, this still returns a 1-D array of zeros.
def get_embeddings_array_from_subtypes(subtypes, embeddings_dict={}, dimension=None):
  if subtypes=="-":
    tokens = []
  else:
    tokens = [t.lower() for t in subtypes.split('-')]
  if dimension is None:
    dimension = embeddings_dict[list(embeddings_dict.keys())[0]].shape[0]
  embedding = np.zeros((len(tokens)+1, dimension))
  for (i,token) in enumerate(tokens):
    embedding[i,:] = get_embedding_from_token(token, embeddings_dict, dimension)
  return embedding

################################################################################
# DATAFRAME PREPROCESSING
################################################################################

# Clean the input card dataframe.
# The returned dataframe contains only relevant columns and no duplicate or irrelevant cards.
# Card information is preprocessed into useful features to be used by the ML model.
def clean_card_data(card_df=None):
  if card_df is None:
    filename = 'cards.csv'
    try:
      card_df = pd.read_csv('cards.csv')
    except:
      card_df = pd.read_csv('drive/MyDrive/MTGNetData/cards.csv')
  card_df = remove_garbage_columns(card_df)
  card_df = remove_duplicate_cards(card_df)
  card_df = remove_reserved_cards(card_df)
  card_df = remove_banned_cards(card_df)
  card_df = remove_transform_cards(card_df)
  return card_df

# Remove any useless columns that are artifacts of combining multiple dataframes into one.
def remove_garbage_columns(card_df):
  try:
    card_df = card_df.drop(columns=['Unnamed: 0'])
  except:
    pass
  try:
    card_df = card_df.drop(columns=["Unnamed: 0.1"])
  except:
    pass
  try:
    card_df = card_df.drop(columns=["Set"])
  except:
    pass
  return card_df

# Remove any cards with fully duplicate rows (as a result of duplicate printings).
def remove_duplicate_cards(card_df):
  card_df = card_df.drop_duplicates(subset=['Name'])
  card_df.reset_index(inplace=True)
  try:
    card_df = card_df.drop(columns=["index"])
  except:
    pass
  return card_df

# Remove from the dataset any cards on the reserved list.
def remove_reserved_cards(card_df, reserved_filename="reserved_list.txt"):
  try:
    with open(reserved_filename) as f:
      reserved_list = [line.replace("\n","").lower() for line in f]
  except:
    return card_df
  card_df = card_df[card_df.apply(lambda row: row.Name.lower() not in reserved_list, axis=1)]
  return card_df

# Remove from the dataset any cards that are banned in commander.
def remove_banned_cards(card_df):
  card_df['Legalities'].fillna("", inplace=True)
  card_df = card_df[card_df.apply(lambda row: "'format': 'commander', 'legality': 'legal'" in row.Legalities.lower(), axis=1)]
  card_df = card_df.drop(columns=["Legalities"])
  return card_df

# TODO: Could remove any cards with the characters "//" in their name since they have additional text on the backside that isn't accounted for.
def remove_transform_cards(card_df):
  return card_df

# Count the amount of a each color of mana in an input mana cost string.
# Input:  mana_cost -- total mana cost string (e.g., {3}{u}{b})
#         symbol -- the symbol to count ('w','u','b','r','g','c','s','x','a')
#           'w' -- amount of white mana
#           'u' -- amount of blue mana
#           'b' -- amount of black mana
#           'r' -- amount of red mana
#           'g' -- amount of green mana
#           'c' -- amount of colorless mana
#           's' -- amount of snow mana ({s})
#           'x' -- 0/1 if the mana cost contains {x}
#           'a' -- amount of generic mana (e.g., {4} --> 4)
# Output: count -- the amount of that type of symbol in the mana cost
# Each colored mana symbol (e.g., {w}) counts as 1.0
# Each hybrid mana symbol (e.g., {w/u}, {2/w}, {w/p}) counts as 0.5 for each color/type
def count_symbols_in_mana_cost(mana_cost, symbol):
  if type(symbol) != str or type(mana_cost) != str:
    return 0.0
  mana_cost, symbol = mana_cost.lower(), symbol.lower()
  all_symbols = ['w','u','b','r','g','c','s','x']
  if symbol not in all_symbols and symbol != 'a':
    return 0.0
  count = 0.0
  count += mana_cost.count('{'+symbol+'}')
  count += 0.5 * mana_cost.count('/'+symbol)
  count += 0.5 * mana_cost.count(symbol+'/')
  if symbol == 'a':
    prev_bracket = 0
    for index, c in enumerate(mana_cost):
      if mana_cost[index] == '{':
          prev_bracket = index
      if index > 0 and index < len(mana_cost)-1:
          if (c not in all_symbols) and mana_cost[index+1] == '}' and ('/' not in mana_cost[prev_bracket+1:index+1]):
              try:
                count += int(mana_cost[prev_bracket+1:index+1])
              except:
                pass
    count += 0.5 * mana_cost.count('{2/')
  return count

# Defines all features that are derived from the non-text statistics of a card.
def non_text_feature_names():
  return ['Power','Toughness','Loyalty','VariablePower','VariableToughness','VariableLoyalty','IsLand','IsCreature','IsArtifact','IsEnchantment','IsPlaneswalker','IsInstant','IsSorcery','IsBattle','IsLegendary','ManaWhite','ManaBlue','ManaBlack','ManaRed','ManaGreen','ManaColorless','ManaSnow','ManaX','ManaGeneric']

# Combine all non-text features from a row in the card dataframe into a single numpy array.
def get_non_text_features_from_row(card_df_row):
  return np.array([card_df_row[col] for col in non_text_feature_names()])

# Defines a condensed set features that are derived from the non-text statistics of a card.
def non_text_feature_names_condensed():
  # Combine 'VariablePower','VariableToughness','VariableLoyalty' into a single 'VariablePTL'
  # Combine 'ManaSnow' and 'ManaColorless' into 'ManaColorlessSnow'
  # Combine 'IsLand'... into 'CondensedType'
  return ['CondensedType','Power','Toughness','Loyalty','VariablePTL','IsLegendary','ManaWhite','ManaBlue','ManaBlack','ManaRed','ManaGreen','ManaColorlessSnow','ManaX','ManaGeneric']

# Combine all condensed non-text features from a row in the card dataframe into a single numpy array.
def get_non_text_features_condensed_from_row(card_df_row):
  return np.array([card_df_row[col] for col in non_text_feature_names_condensed()])

# Return an integer describing the card type combination of the input row in the card dataframe.
# All possible combinations of card types are represented by integers 0-16.
def get_condensed_card_type(card_df_row):
  if card_df_row['IsLand'] and card_df_row['IsArtifact'] and card_df_row['IsCreature']:
    return 16
  elif card_df_row['IsLand'] and card_df_row['IsArtifact'] and card_df_row['IsEnchantment']:
    return 15
  elif card_df_row['IsCreature'] and card_df_row['IsArtifact'] and card_df_row['IsEnchantment']:
    return 14
  elif card_df_row['IsLand'] and card_df_row['IsArtifact']:
    return 13
  elif card_df_row['IsLand'] and card_df_row['IsCreature']:
    return 12
  elif card_df_row['IsLand'] and card_df_row['IsEnchantment']:
    return 11
  elif card_df_row['IsEnchantment'] and card_df_row['IsArtifact']:
    return 10
  elif card_df_row['IsEnchantment'] and card_df_row['IsCreature']:
    return 9
  elif card_df_row['IsArtifact'] and card_df_row['IsCreature']:
    return 8
  elif card_df_row['IsArtifact']:
    return 7
  elif card_df_row['IsEnchantment']:
    return 6
  elif card_df_row['IsLand']:
    return 5
  elif card_df_row['IsCreature']:
    return 4
  elif card_df_row['IsPlaneswalker']:
    return 3
  elif card_df_row['IsInstant']:
    return 2
  elif card_df_row['IsSorcery']:
    return 1
  elif card_df_row['IsBattle']:
    return 0
  return 0

# Get the price of a single input card name. Returns 0 if fetcher fails to get the price.
def get_card_price(card_name):
  try:
    card_price = fetcher.get(card_name)
  except:
    card_price = 0.0
  return card_price

# Defines all price categories:
def price_categories():
  price_limits = [7.5,       3,      1,        0.15,  0]
  price_labels = ["extreme", "high", "medium", "low", "budget"]
  price_ints   = [4,         3,      2,        1,     0]
  return zip(price_limits, price_labels, price_ints)

# Returns which category the input price falls into.
def get_price_category(price):
  for (limit, label, intlabel) in price_categories():
    if price >= limit:
      return intlabel
  return 0

# Build a vocabulary of ngrams (1-3) and turn them into feature vectors for each card's text.
def get_ngram_features(card_text_list, max_N=2):
  if type(card_text_list) != list:
    card_text_list = list(card_text_list)
  stemmer = PorterStemmer()
  card_text_list = [" ".join([stemmer.stem(t) for t in split_text_into_tokens(card_text)]) for card_text in card_text_list]
  ngrams_list, ngram_features_list = [], []
  ngram_vocabulary = {}
  voc_i = 0
  for this_text in card_text_list:
    tokens = split_text_into_tokens(this_text)
    this_ngrams = []
    for ni in range(1,max_N+1):
      this_ngrams = this_ngrams + list(ngrams(tokens, ni))
    ngrams_list.append(this_ngrams)
    for this_ngram in this_ngrams:
      if tuple(this_ngram) not in ngram_vocabulary.keys():
        ngram_vocabulary[tuple(this_ngram)] = voc_i
        voc_i += 1
  for this_ngrams in ngrams_list:
    feature_array = np.zeros((len(this_ngrams), len(ngram_vocabulary)))
    for i,this_ngram in enumerate(this_ngrams):
      feature_array[i, ngram_vocabulary[tuple(this_ngram)]] = 1
    ngram_features_list.append(feature_array)
  print("NGRAM VOCAB SIZE...")
  print(ngram_features_list[0].shape)
  return ngram_features_list

# Create a dataframe with all features relevant to the ML model from the card statistics.
def get_feature_df(card_df, initialize=True, verbose=True, expanded=False):
  card_df = card_df.copy()
  # Get the price of each card.
  if verbose:
    print("Loading price data from csv...")
  price_filename = 'card_prices.csv'
  try:
    try:
        price_df = pd.read_csv(price_filename)
    except:
        price_df = pd.read_csv('drive/MyDrive/MTGNetData/'+price_filename)
    card_df['Price'] = card_df.apply(lambda row: float(get_card_price_from_price_dataframe(row.Name, price_df)), axis=1)
  except:
    print("Failed to load price data. Re-fetching data with fetcher...")
    card_df['Price'] = card_df.apply(lambda row: float(get_card_price(row.Name)), axis=1)
  # Get the price category of each card.
  card_df["PriceCategory"] = card_df.apply(lambda row: get_price_category(row.Price), axis=1)
  # Preprocess rules text (replace instances of symbols with text descriptions)
  if verbose:
    print("Processing rules text...")
  # Fill text fields with defaults.
  card_df['SubTypes'].fillna("", inplace=True)
  card_df['Text'].fillna("", inplace=True)
  card_df['Text'] =            card_df.apply(lambda row: preprocess_card_text(row.Name, row.Text), axis=1)
  # Build a vocabulary for every word in all cards:
  vocab_dimension = 50
  if initialize:
    all_text = " ".join(list(card_df['Text']))
    if verbose:
      print("Building embedding vocabulary...")
    embeddings_dict = build_embedding_vocabulary(all_text, dimension=vocab_dimension)
  else:
    embeddings_dict = {}
  if verbose:
    print("Processing non-text attributes...")
  # Adjust power, toughness, loyalty. Account for X or * in power/toughness/loyalty.
  card_df['Power'].fillna(-1.0, inplace=True)
  card_df['Toughness'].fillna(-1.0, inplace=True)
  card_df['Loyalty'].fillna(-1.0, inplace=True)
  card_df['VariablePower'] =     card_df.apply(lambda row: 1.0 if (type(row.Power)==str and     (row.Power.lower()=="x"     or "*" in row.Power.lower()))     else 0.0, axis=1)
  card_df['VariableToughness'] = card_df.apply(lambda row: 1.0 if (type(row.Toughness)==str and (row.Toughness.lower()=="x" or "*" in row.Toughness.lower())) else 0.0, axis=1)
  card_df['VariableLoyalty'] =   card_df.apply(lambda row: 1.0 if (type(row.Loyalty)==str and   (row.Loyalty.lower()=="x"   or "*" in row.Loyalty.lower()))   else 0.0, axis=1)
  card_df['Power'] =             card_df.apply(lambda row: float(row.Power)     if ((type(row.Power)==str     and not (row.Power.lower()=="x"     or "*" in row.Power.lower()))     or row.Power==-1)     else 0.0 + ("1" in row.Power), axis=1)
  card_df['Toughness'] =         card_df.apply(lambda row: float(row.Toughness) if ((type(row.Toughness)==str and not (row.Toughness.lower()=="x" or "*" in row.Toughness.lower())) or row.Toughness==-1) else 0.0 + ("1" in row.Toughness), axis=1)
  card_df['Loyalty'] =           card_df.apply(lambda row: float(row.Loyalty)   if ((type(row.Loyalty)==str   and not (row.Loyalty.lower()=="x"   or "*" in row.Loyalty.lower()))   or row.Loyalty==-1)   else 0.0 + ("1" in row.Loyalty), axis=1)
  # Adjust types (replace list of types with boolean variables for each possible type).
  card_df['IsLand'] =         card_df.apply(lambda row: 1.0 if "land"         in row.CardTypes.lower()  else 0.0, axis=1)
  card_df['IsCreature'] =     card_df.apply(lambda row: 1.0 if "creature"     in row.CardTypes.lower()  else 0.0, axis=1)
  card_df['IsArtifact'] =     card_df.apply(lambda row: 1.0 if "artifact"     in row.CardTypes.lower()  else 0.0, axis=1)
  card_df['IsEnchantment'] =  card_df.apply(lambda row: 1.0 if "enchantment"  in row.CardTypes.lower()  else 0.0, axis=1)
  card_df['IsPlaneswalker'] = card_df.apply(lambda row: 1.0 if "planeswalker" in row.CardTypes.lower()  else 0.0, axis=1)
  card_df['IsInstant'] =      card_df.apply(lambda row: 1.0 if "instant"      in row.CardTypes.lower()  else 0.0, axis=1)
  card_df['IsSorcery'] =      card_df.apply(lambda row: 1.0 if "sorcery"      in row.CardTypes.lower()  else 0.0, axis=1)
  card_df['IsBattle'] =       card_df.apply(lambda row: 1.0 if "battle"       in row.CardTypes.lower()  else 0.0, axis=1)
  card_df['IsLegendary'] =    card_df.apply(lambda row: 1.0 if "legendary"    in row.SuperTypes.lower() else 0.0, axis=1) # Other supertypes we're ignoring: Basic, Snow, World
  card_df.drop(columns=["SuperTypes"], inplace=True)
  # Rearrange column list:
  card_df = card_df[['Name','Price','PriceCategory','CardTypes','Power','Toughness','Loyalty','VariablePower','VariableToughness','VariableLoyalty','IsLand','IsCreature','IsArtifact','IsEnchantment','IsPlaneswalker','IsInstant','IsSorcery','IsBattle','IsLegendary','SubTypes','Text','ManaCost']]
  # Adjust mana cost (replace the symbols representing the mana cost with numeric feature vectors).
  card_df['ManaWhite'] =      card_df.apply(lambda row: count_symbols_in_mana_cost(row.ManaCost,'w'), axis=1)
  card_df['ManaBlue'] =       card_df.apply(lambda row: count_symbols_in_mana_cost(row.ManaCost,'u'), axis=1)
  card_df['ManaBlack'] =      card_df.apply(lambda row: count_symbols_in_mana_cost(row.ManaCost,'b'), axis=1)
  card_df['ManaRed'] =        card_df.apply(lambda row: count_symbols_in_mana_cost(row.ManaCost,'r'), axis=1)
  card_df['ManaGreen'] =      card_df.apply(lambda row: count_symbols_in_mana_cost(row.ManaCost,'g'), axis=1)
  card_df['ManaColorless'] =  card_df.apply(lambda row: count_symbols_in_mana_cost(row.ManaCost,'c'), axis=1)
  card_df['ManaSnow'] =       card_df.apply(lambda row: count_symbols_in_mana_cost(row.ManaCost,'s'), axis=1)
  card_df['ManaX'] =          card_df.apply(lambda row: count_symbols_in_mana_cost(row.ManaCost,'x'), axis=1)
  card_df['ManaGeneric'] =    card_df.apply(lambda row: count_symbols_in_mana_cost(row.ManaCost,'a'), axis=1)
  card_df.drop(columns=["ManaCost"], inplace=True)
  # Add condensed non-text features:
  # Combine 'VariablePower','VariableToughness','VariableLoyalty' into a single 'VariablePTL'
  card_df['VariablePTL'] =        card_df.apply(lambda row: max(row.VariablePower, row.VariableToughness, row.VariableLoyalty), axis=1)
  # Combine 'ManaSnow' and 'ManaColorless' into 'ManaColorlessSnow'
  card_df['ManaColorlessSnow'] =  card_df.apply(lambda row: row.ManaSnow+row.ManaColorless, axis=1)
  # Combine 'IsLand'... into 'CondensedType'
  card_df['CondensedType'] =      card_df.apply(lambda row: get_condensed_card_type(row), axis=1)
  # Permute rows with newlines into all possible combinations of those rows:
  if expanded:
      if verbose:
        print("Adding all permutations of cards with newlines...")
      card_df = card_df.reset_index()
      card_df["Permuted"] = card_df.apply(lambda row: 0, axis=1)
      permuted_dataframes = [card_df]
      max_permutations = 3 # Maximum number of permutations of a single card allowed
      for ni, name in enumerate(card_df["Name"]):
        this_df = card_df[card_df["Name"]==name]
        this_df = this_df.reset_index()
        text_permutations = newline_permutations(this_df.loc[0].Text)
        if len(text_permutations)>1:
          card_df.loc[ni, "Permuted"] = 1
          this_df["Permuted"] = 2
          df_copies = pd.DataFrame(np.repeat(this_df.values, min(len(text_permutations)-1, max_permutations+this_df.loc[0].PriceCategory), axis=0))
          df_copies.columns = this_df.columns
          surplus_permutations = np.array(text_permutations[1:])
          np.random.shuffle(surplus_permutations)
          new_permutations = list(surplus_permutations)
          n = 0
          for j,row in df_copies.iterrows():
            df_copies.loc[j, "Text"] = new_permutations[n]
            n += 1
          permuted_dataframes.append(df_copies)
      card_df = pd.concat(permuted_dataframes)
      try:
        card_df = card_df.drop(columns=["level_0"])
      except:
        pass
      card_df = card_df.reset_index()
      try: 
        card_df = card_df.drop(columns=["index","level_0"])
      except:
        pass
  card_df['Text'] = card_df.apply(lambda row: clean_newlines(row.Text), axis=1)
  if verbose:
    print("Creating feature arrays...")
  # Replace all non-text features with numpy arrays.
  card_df['NonTextFeatures'] =          card_df.apply(lambda row: get_non_text_features_from_row(row), axis=1)
  card_df['NonTextFeaturesCondensed'] = card_df.apply(lambda row: get_non_text_features_condensed_from_row(row), axis=1)
  card_df.drop(columns = non_text_feature_names(), inplace=True)
  card_df.drop(columns = ['VariablePTL', 'ManaColorlessSnow', 'CondensedType'], inplace=True)
  # Scale non-text features and prices.
  card_df = scale_non_text_features(card_df, fit_scaler=initialize)
  card_df = scale_condensed_non_text_features(card_df, fit_scaler=initialize)
  # Replace the subtypes and the card text with arrays of glove features.
  card_df['SubTypeFeatures'] = card_df.apply(lambda row: get_embeddings_array_from_subtypes(row.SubTypes, embeddings_dict=embeddings_dict, dimension=vocab_dimension), axis=1)
  card_df['TextFeatures'] =    card_df.apply(lambda row: get_embeddings_array_from_text(row.Text, embeddings_dict=embeddings_dict, dimension=vocab_dimension), axis=1)
  # Get ngram feature vectors:
  # card_df['NGramFeatures'] = get_ngram_features(card_df['Text'])
  card_df.drop(columns=["Text","SubTypes"], inplace=True)
  # Reset the index to 0-N.
  card_df.reset_index(inplace=True)
  card_df.drop(columns=["index"], inplace=True)
  return card_df

# Returns a modified version of the card dataframe where each feature is scaled to a (-1,1) range.
def scale_non_text_features(card_feature_df, fit_scaler=True):
  card_feature_df = card_feature_df.copy()
  nontext_features = get_non_text_feature_array(card_feature_df)
  global NONTEXT_SCALER
  if fit_scaler:
    NONTEXT_SCALER.fit(nontext_features)
  card_feature_df['NonTextFeatures'] = card_feature_df.apply(lambda row: NONTEXT_SCALER.transform(row.NonTextFeatures.reshape(1, -1)).reshape((len(non_text_feature_names()),)), axis=1)
  return card_feature_df

# Returns a modified version of the card dataframe where each feature is scaled to a (-1,1) range.
def scale_condensed_non_text_features(card_feature_df, fit_scaler=True):
  card_feature_df = card_feature_df.copy()
  nontext_features = get_condensed_non_text_feature_array(card_feature_df)
  global NONTEXT_CONDENSED_SCALER
  if fit_scaler:
    NONTEXT_CONDENSED_SCALER.fit(nontext_features)
  card_feature_df['NonTextFeaturesCondensed'] = card_feature_df.apply(lambda row: NONTEXT_CONDENSED_SCALER.transform(row.NonTextFeaturesCondensed.reshape(1, -1)).reshape((len(non_text_feature_names_condensed()),)), axis=1)
  return card_feature_df

# Returns a modified version of the card dataframe where each price is scaled to a N(0,1) range (StandardScaler to better handle outliers)
def scale_prices(card_feature_df):
  card_feature_df = card_feature_df.copy()
  price_array = get_price_array(card_feature_df)
  global PRICE_SCALER
  PRICE_SCALER.fit(price_array)
  card_feature_df['Price'] = card_feature_df.apply(lambda row: PRICE_SCALER.transform(np.array([row.Price]).reshape(1, -1)), axis=1)
  return card_feature_df

# Returns an Nx24 numpy array with all non-text features from the input dataframe.
# Each row is the non-text features of a single card.
def get_non_text_feature_array(card_feature_df):
  return np.concatenate([f.reshape(1,f.shape[0]) for f in card_feature_df["NonTextFeatures"].values.tolist()])

# Returns an Nx14 numpy array with all condensed non-text features from the input dataframe.
# Each row is the condensed non-text features of a single card.
def get_condensed_non_text_feature_array(card_feature_df):
  return np.concatenate([f.reshape(1,f.shape[0]) for f in card_feature_df["NonTextFeaturesCondensed"].values.tolist()])

# Returns a list with all subtype and text features from the input dataframe.
# Each element is all subtype and text features for a single card.
def get_all_text_feature_array(card_feature_df):
  return [np.concatenate([sub, text]) for (sub,text) in zip(card_feature_df["SubTypeFeatures"].values.tolist(), card_feature_df["TextFeatures"].values.tolist())]

# Returns a list with all text features from the input dataframe.
# Each element is all text features for a single card.
def get_text_only_feature_array(card_feature_df):
  return card_feature_df["TextFeatures"].values.tolist()

# Returns a list with all ngram text features from the input dataframe.
# Each element is all ngram text features for a single card.
def get_ngram_feature_array(card_feature_df):
  return card_feature_df["NGramFeatures"].values.tolist()

# Returns an N-dimension numpy array of price features.
def get_price_array(card_feature_df):
  price = np.array([f for f in card_feature_df["Price"].values.tolist()])
  return price.reshape((price.shape[0], 1))

# Returns an N-dimension numpy array of price category features.
def get_price_category_array(card_feature_df):
  price = np.array([f for f in card_feature_df["PriceCategory"].values.tolist()])
  return price.reshape((price.shape[0], 1))

# Re-evaluates the price category for each price in the card feature dataframe.
# Used to quickly change the price category definitions without re-computing the feature dataframe.
def reassign_price_category(card_feature_df):
  card_feature_df["PriceCategory"] = card_feature_df.apply(lambda row: get_price_category(row.Price), axis=1)
  return card_feature_df

# Save the card feature dataframe to the input filename for quick re-loading and analysis.
def save_card_feature_df(card_feature_df, filename="card_features.csv", expanded=False):
  if expanded:
    filename = filename.replace(".csv", "_expanded.csv")
  print("Saving card feature dataframe...")
  saved_df = card_feature_df.copy()
  saved_df["SubTypeFeatureRows"] = saved_df.apply(lambda row: row.SubTypeFeatures.shape[0], axis=1)
  saved_df["TextFeatureRows"] =    saved_df.apply(lambda row: row.TextFeatures.shape[0], axis=1)
  # saved_df["NGramFeatureRows"] =   saved_df.apply(lambda row: row.NGramFeatures.shape[0], axis=1)
  nontext_feature_array = get_non_text_feature_array(saved_df)
  condensed_nontext_feature_array = get_condensed_non_text_feature_array(saved_df)
  subtype_feature_array = np.concatenate([f for f in saved_df["SubTypeFeatures"].values.tolist()])
  text_feature_array = np.concatenate([f for f in saved_df["TextFeatures"].values.tolist()])
  # ngram_feature_array = np.concatenate([n for n in saved_df["NGramFeatures"].values.tolist()])
  saved_df.drop(columns=["NonTextFeatures","NonTextFeaturesCondensed","TextFeatures","SubTypeFeatures"], inplace=True)
  saved_df.to_csv(filename)
  np.savez(filename.replace(".csv","_numpy.npz"), nontext_feature_array, condensed_nontext_feature_array, subtype_feature_array, text_feature_array)
  return saved_df

# Load the card feature dataframe from the specified filename.
def load_card_feature_df(filename="card_features.csv", expanded=False):
  if expanded:
    filename = filename.replace(".csv", "_expanded.csv")
  try:
    loaded_df = pd.read_csv(filename)
    loaded_arrs = np.load(filename.replace(".csv","_numpy.npz"))
  except:
    filename = "drive/MyDrive/MTGNetData/"+filename
    loaded_df = pd.read_csv(filename)
    loaded_arrs = np.load(filename.replace(".csv","_numpy.npz"))
  nontext_arr, condensed_nontext_arr, subtype_arr, text_arr = loaded_arrs["arr_0"], loaded_arrs["arr_1"], loaded_arrs["arr_2"], loaded_arrs["arr_3"]
  loaded_df["NonTextFeatures"] = ''
  loaded_df["NonTextFeaturesCondensed"] = ''
  loaded_df["SubTypeFeatures"] = ''
  loaded_df["TextFeatures"] = ''
  subtype_i, text_i, ngram_i = 0, 0, 0
  for i,row in loaded_df.iterrows():
    loaded_df.at[i, "NonTextFeatures"] = nontext_arr[i,:]
    loaded_df.at[i, "NonTextFeaturesCondensed"] = condensed_nontext_arr[i,:]
    loaded_df.at[i, "SubTypeFeatures"] = subtype_arr[subtype_i:subtype_i+loaded_df.loc[i, "SubTypeFeatureRows"], :]
    loaded_df.at[i, "TextFeatures"] =    text_arr[text_i:text_i+loaded_df.loc[i, "TextFeatureRows"], :]
    # loaded_df.at[i, "NGramFeatures"] =   ngram_arr[ngram_i:ngram_i+loaded_df.loc[i, "NGramFeatureRows"], :]
    subtype_i += loaded_df.loc[i, "SubTypeFeatureRows"]
    text_i += loaded_df.loc[i, "TextFeatureRows"]
    # ngram_i += loaded_df.loc[i, "NGramFeatureRows"]
  try:
    loaded_df.drop(columns=["TextFeatureRows","SubTypeFeatureRows"], inplace=True)
  except:
    pass
  try:
    loaded_df.drop(columns=["Unnamed: 0"], inplace=True)
  except:
    pass
  if expanded:
    loaded_df = loaded_df[["Name", "Permuted", "CardTypes", "NonTextFeatures", "NonTextFeaturesCondensed", "SubTypeFeatures", "TextFeatures", "Price", "PriceCategory"]]
  else:
    loaded_df = loaded_df[["Name", "CardTypes", "NonTextFeatures", "NonTextFeaturesCondensed", "SubTypeFeatures", "TextFeatures", "Price", "PriceCategory"]]
  return loaded_df

# Oversample from under-represented price categories until the dataframe contains the same number of samples for each category.
# If undersample=True, then the most frequent class is first undersampled until it contains as many samples as the second most frequent class.
# In all cases, all other classes are then oversampled until they contain the same number of samples as the majority class.
def balance_training_dataset(card_feature_df, random_state=1, undersample=False):
  if random_state is not None:
    np.random.seed(random_state)
  if undersample:
    card_feature_df = card_feature_df.reset_index()
    price_groupings = card_feature_df.groupby(["PriceCategory"]).count()["Name"]
    max_class, max_samples = np.argmax(price_groupings), np.max(price_groupings)
    target_samples = np.sort(price_groupings)[-2]
    df_max_class = card_feature_df[card_feature_df["PriceCategory"]==max_class]
    indices_to_sample = np.random.choice(df_max_class.index.tolist(), size=target_samples, replace=False).tolist()
    card_feature_df = pd.concat([card_feature_df.loc[indices_to_sample], card_feature_df[card_feature_df["PriceCategory"]!=max_class]])
  card_feature_df = card_feature_df.reset_index()
  price_groupings = card_feature_df.groupby(["PriceCategory"]).count()["Name"]
  target_samples = np.max(price_groupings)
  resampled_dfs = [card_feature_df]
  for pclass, samples_this_class in enumerate(price_groupings):
    oversamples = target_samples - samples_this_class
    df_this_class = card_feature_df[card_feature_df["PriceCategory"]==pclass]
    if oversamples>0:
      indices_to_resample = np.random.choice(df_this_class.index.tolist(), size=oversamples, replace=True).tolist()
      resampled_dfs.append(card_feature_df.loc[indices_to_resample])
  card_feature_df = pd.concat(resampled_dfs)
  return card_feature_df

# Save the feature scalers for nontext, nontext condensed, and text features:
def save_scalers(expanded=False):
  if expanded:
    joblib.dump(NONTEXT_SCALER, 'nontext_scaler_expanded.gz')
    joblib.dump(NONTEXT_CONDENSED_SCALER, 'nontext_condensed_scaler_expanded.gz')
    joblib.dump(TEXT_SCALER, 'text_scaler_expanded.gz')
  else:
    joblib.dump(NONTEXT_SCALER, 'nontext_scaler.gz')
    joblib.dump(NONTEXT_CONDENSED_SCALER, 'nontext_condensed_scaler.gz')
    joblib.dump(TEXT_SCALER, 'text_scaler.gz')

# Reload the saved scalers for nontext, nontext condensed, and text features:
def load_scalers(expanded=False):
  global NONTEXT_SCALER
  global NONTEXT_CONDENSED_SCALER
  global TEXT_SCALER
  if expanded:
    nontext_filename = 'nontext_scaler_expanded.gz'
    nontext_condensed_filename = 'nontext_condensed_scaler_expanded.gz'
    text_filename = 'text_scaler_expanded.gz'
  else:
    nontext_filename = 'nontext_scaler.gz'
    nontext_condensed_filename = 'nontext_condensed_scaler.gz'
    text_filename = 'text_scaler.gz'
  try:
    NONTEXT_SCALER = joblib.load(nontext_filename)
  except:
    NONTEXT_SCALER = joblib.load("drive/MyDrive/MTGNetData/"+nontext_filename)
  try:
    NONTEXT_CONDENSED_SCALER = joblib.load(nontext_condensed_filename)
  except:
    NONTEXT_CONDENSED_SCALER = joblib.load("drive/MyDrive/MTGNetData/"+nontext_condensed_filename)
  try:
    TEXT_SCALER = joblib.load(text_filename)
  except:
    TEXT_SCALER = joblib.load('drive/MyDrive/MTGNetData/'+text_filename)

##############################################################################################
# MODEL UTILITY FUNCTIONS --------------------------------------------------------------------
##############################################################################################

# Splits the input model data (x, y) into training, testing, validation data.
# Inputs: x, y -- the model inputs/outputs
#         test_ratio  -- a 0-1 ratio representing how much of the input data should be used for testing
#         valid_ratio -- a 0-1 ratio representing how much of the input data should be used for validation
def split_model_data(x, y, test_ratio=0.2, valid_ratio=0.15, random_state=1):
  if test_ratio+valid_ratio >= 1:
    test_ratio, valid_ratio = 0.2, 0.15
  if (type(x)==list and len(x)==0) or (type(y)==list and len(y)==0):
    xtrain, xvalid, xtest, ytrain, yvalid, ytest = [],[],[],[],[],[]
  else:
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=test_ratio, random_state=random_state)
    xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, test_size=(valid_ratio / (1-test_ratio)), random_state=random_state)
  return xtrain, xvalid, xtest, ytrain, yvalid, ytest

# Splits the input dataframe into three: training, testing, and validation dataframes.
# Inputs: card_feature_df -- a pandas dataframe with all card features
#         test_ratio  -- a 0-1 ratio representing how much of the input data should be used for testing
#         valid_ratio -- a 0-1 ratio representing how much of the input data should be used for validation
#         split_permutations -- True if permuted cards can be included in the validation and test sets, False (default) if they should only be used in training
def split_dataframe(card_feature_df, test_ratio=0.2, valid_ratio=0.15, random_state=None, split_permutations=False):
  if test_ratio+valid_ratio >= 1:
    test_ratio, valid_ratio = 0.2, 0.15
  if "Permuted" not in card_feature_df.columns:
    split_permutations = True
  if not split_permutations:
    permuted_df = card_feature_df[card_feature_df["Permuted"]==2]
    card_feature_df = card_feature_df[card_feature_df["Permuted"]<=1]
  train_inds, test_inds =  train_test_split(list(range(len(card_feature_df.index))), test_size=test_ratio, random_state=random_state)
  train_inds, valid_inds = train_test_split(train_inds, test_size=(valid_ratio / (1-test_ratio)), random_state=random_state)
  train_df, valid_df, test_df = card_feature_df.iloc[train_inds], card_feature_df.iloc[valid_inds], card_feature_df.iloc[test_inds]
  train_df, valid_df, test_df = train_df.reset_index(), valid_df.reset_index(), test_df.reset_index()
  train_df, valid_df, test_df = train_df.drop(columns=["index"]), valid_df.drop(columns=["index"]), test_df.drop(columns=["index"])
  if not split_permutations:
    permuted_df = permuted_df.reset_index(drop=True)
    train_mask, valid_mask, test_mask = [], [], []
    for i,row in permuted_df.iterrows():
        if row.Name in train_df["Name"].tolist():
            train_mask.append(i)
        elif row.Name in valid_df["Name"].tolist():
            valid_mask.append(i)
        elif row.Name in test_df["Name"].tolist():
            test_mask.append(i)
    valid_perm_df = permuted_df.iloc[valid_mask]
    test_perm_df = permuted_df.iloc[test_mask]
    permuted_df = permuted_df.iloc[train_mask]
    train_df = pd.concat([train_df, permuted_df], ignore_index=True)
    train_df = train_df.sample(frac=1).reset_index(drop=True)
  else:
    valid_perm_df = []
    test_perm_df = []
  return train_df, (valid_df, valid_perm_df), (test_df, test_perm_df)

# Returns the mean squared error for the input true and predicted price data.
def get_mse(ytrue, ypred):
  if type(ytrue)==list:
    ytrue = np.array(ytrue)
  if type(ypred)==list:
    ypred = np.array(ypred)
  return np.square(ytrue - ypred).mean()

# For the input array, convert each element from an array of probabilities to 
# an integer giving the category with the highest probability.
# Input: ypred -- an NxC numpy array with N samples and probabilities for each of C classes  
def convert_price_category_arrays_to_ints(ypred):
  if type(ypred)==list:
    ypred = np.concatenate(ypred, axis=0)
  ypred_ints = np.zeros((ypred.shape[0],1))
  for i in range(ypred.shape[0]):
    ypred_ints[i] = np.argmax(ypred[i,:])
  return ypred_ints

# For the input array, convert each element from an integer giving the price
# category to an array giving probabilities for each category.
# (Predicted category gets probability 1, others get probability 0.)
# Input: ypred -- a list or numpy array of integers giving prediction categories
def convert_price_category_ints_to_arrays(ypred):
  ypred = list(ypred)
  if type(ypred[0])==np.ndarray:
    ypred = [y[0] for y in ypred]
  ypred_arrays= np.zeros((len(ypred), len(list(price_categories()))))
  for i,this_y in enumerate(ypred):
    ypred_arrays[i,this_y] = 1
  return ypred_arrays

# Evaluates precision, recall, and f1 score by class.
def classifier_metrics(y, ypred):
  evals = metrics.precision_recall_fscore_support(y, ypred, average=None, labels=list(range(len(list(price_categories())))))
  precision, recall, fscore, support = [list(ev) for ev in evals]
  return precision, recall, fscore, support

# Returns a smaller proportion of the available data:
def downselect(x, y, ratio=0.1):
  if ratio==1:
    return x,y
  if type(x)==list:
    input_size = len(x)
  elif type(x)==np.ndarray:
    input_size = x.shape[0]
  else:
    return x,y
  samples = round(input_size*ratio)
  sampled_indices = np.random.choice(np.arange(input_size), size=samples, replace=False)
  if type(x)==list:
    return [x[s] for s in sampled_indices], [y[s] for s in sampled_indices]
  elif type(x)==np.ndarray:
    return x[sampled_indices, :], y[sampled_indices, :]

##############################################################################################
# MODEL DEFINITIONS --------------------------------------------------------------------------
##############################################################################################

################################################################################
# Baseline 1 (most basic) -- Frequency Model
# For a given card, predicts its price category by returning the most frequent price category for that card's type.
# If the card has multiple types, returns the an "average" prediction from each type.
class FrequencyModel:

  # Initialize AverageModel by simply listing all card types.
  def __init__(self):
    self.card_types = ['Land', 'Creature', 'Artifact', 'Enchantment', 'Planeswalker', 'Instant', 'Sorcery', 'Battle']
    self.card_type_prices = {k:0.0 for k in self.card_types}
    self.trained = False

  def __str__(self):
    return "FrequencyModel"
  
  # Gets model data in the proper format for the FrequencyModel.
  # Outputs:  x -- N-dim list of card types
  #           y -- N-dim list of card price categories
  def get_model_data(self, card_feature_df):
    return card_feature_df["CardTypes"].values.tolist(), card_feature_df["PriceCategory"].values.tolist()

  # FrequencyModel learns the most frequent price of each card type.
  def train(self, xtrain=[], ytrain=[], verbose=True):
    if verbose:
      print("Training FrequencyModel...")
    for card_type in self.card_types:
      this_card_type_prices = [ytrain[i] for i in range(len(ytrain)) if card_type in xtrain[i]]
      if len(this_card_type_prices)==0:
        self.card_type_prices[card_type] = 0
      else:
        self.card_type_prices[card_type] = np.bincount(this_card_type_prices).argmax()
    self.trained = True

  # Predict output features (price category) for the given input features (card type).
  def predict(self, x):
    types_split = [i.split('-') for i in x]
    ypred = []
    for i,_ in enumerate(types_split):
      # If we don't have a card type at all (should not happen):
      if len(types_split[i])==2 and len(types_split[i][0])==0 and len(types_split[i][1])==0:
        ypred.append(0)
      # If there is a single card type, return the most frequent category for that type:
      elif len(types_split[i])==1:
        try:
          ypred.append(self.card_type_prices[types_split[i][0]])
        except:
          ypred.append(0)
      # If there are multiple card types, average the most frequent categories across each type:
      else:
        mean_price = 0
        for j in range(len(types_split[i])):
          try:
            mean_price += self.card_type_prices[types_split[i][j]]
          except:
            mean_price += 0
        ypred.append(round(mean_price / len(types_split[i])))
    return ypred

  # Predict output features (price category) and evaluate model accuracy for the given input features (card type).
  def evaluate(self, x, y):
    ypred = self.predict(x)
    ypred_adjusted = convert_price_category_ints_to_arrays(ypred)
    loss = keras.losses.sparse_categorical_crossentropy(y, ypred_adjusted)
    loss = loss.numpy().mean()
    accuracy = keras.metrics.Accuracy()
    accuracy.update_state(y, ypred)
    accuracy = accuracy.result().numpy()
    precision, recall, fscore, support = classifier_metrics(y, ypred)
    return {"sparse_categorical_crossentropy":loss, "accuracy":accuracy, "precision":precision, "recall":recall, "fscore":fscore, "support":support}

################################################################################
# Baseline 2 -- Card Statistics Model
# Uses a neural network to predict price category from all non-text card features.
# (e.g., power, toughness, mana statistics, supertypes)
class StatsModel:

  # Initialize StatsModel by building a basic neural network.
  def __init__(self, num_layers=2, num_units=128, condensed=True, verbose=False):
    if condensed:
      model_input_shape = len(non_text_feature_names_condensed())
    else:
      model_input_shape = len(non_text_feature_names())
    model = keras.Sequential()
    model.add(layers.Input(shape=(model_input_shape,)))
    for i in range(num_layers):
      model.add(layers.Dense(num_units, activation="relu"))
    model.add(layers.Dense(len(list(price_categories())), activation="softmax"))
    if verbose:
      model.summary()
    self.model = model
    self.condensed = condensed
    self.trained = False

  def __str__(self):
    return "StatsModel"

  # Gets model data in the proper format for the StatsModel.
  # Outputs:  x -- Nx24 (or Nx14 if condensed) numpy array of non-text card features
  #           y -- Nx1  numpy array of card prices
  def get_model_data(self, card_feature_df):
    if self.condensed:
      x = get_condensed_non_text_feature_array(card_feature_df)
    else:
      x = get_non_text_feature_array(card_feature_df)
    y = get_price_category_array(card_feature_df)
    return x, y
  
  # Train the neural net to predict price from textless features.
  def train(self, xtrain, ytrain, loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'], batch_size=256, epochs=50, verbose=2):
    if verbose:
      print("Training StatsModel...")
    self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    self.model.fit(xtrain, ytrain, batch_size=batch_size, epochs=epochs, verbose=verbose)
    self.trained = True

  # Predict output features (price category) for the given input features (non-text features).
  def predict(self, x, verbose=0):
    return self.model.predict(x, verbose=verbose)

  # Predict output features (price category) and evaluate model accuracy for the given input features (non-text features).
  def evaluate(self, x, y, verbose=0):
    results = self.model.evaluate(x, y, verbose=verbose)
    loss, accuracy = results[0], results[1]
    precision, recall, fscore, support = classifier_metrics(y, convert_price_category_arrays_to_ints(self.predict(x)))
    return {"sparse_categorical_crossentropy":loss, "accuracy":accuracy, "precision":precision, "recall":recall, "fscore":fscore, "support":support}

################################################################################
# NLP-Based Text Model Allowing Variable Length Inputs
# Uses an LSTM with all the textless/card statistics features of the StatsModel, but also 
# uses word vectors for each word in the card text and each subtype.
# Model inputs can be of any shape -- i.e., no padding is ussed on the inputs.
class ShapelessTextModel:

  # Initialize ShapelessTextModel by building a stateful LSTM.
  # Choose whether to include subtype features along with rules text features by toggling include_subtype_as_features (default False)
  def __init__(self, num_layers=2, num_units=128, dimension=50, verbose=False, include_subtype_as_features=False):
    model = keras.Sequential()
    model.add(layers.LSTM(num_units, activation='relu', stateful=True, batch_input_shape=(1, None, dimension), return_sequences=True)) 
      # use batch_input_shape=(batch_size, sequence_length, input_size) with batch_size=1
      # set stateful=True for one-by-one model training
    for i in range(num_layers-1):
      model.add(layers.LSTM(num_units, activation='relu')) # specify only number of output units
    model.add(layers.Dense(len(list(price_categories())), activation="softmax"))
    if verbose:
      model.summary()
    self.model = model
    self.include_subtype_as_features = include_subtype_as_features
    self.trained = False

  def __str__(self):
    return "ShapelessTextModel" + self.condensed*" (Condensed)"

  # Ensure no NANs in the dataset:
  def clean_nan_data(self, x_list, y_list):
    x_list_clean, y_list_clean = [], []
    for i in range(len(x_list)):
      this_x, this_y = x_list[i], y_list[i]
      found_nan = np.isnan(this_x).any() or np.isnan(this_y).any()
      if not found_nan:
        x_list_clean.append(this_x)
        y_list_clean.append(this_y)
    return x_list_clean, y_list_clean

  # Gets model data in the proper format for the ShapelessTextModel.
  # Outputs:  x -- N-dim list where each element is a numpy array card features
  #           y -- N-dim list where each element is a price category (int)
  def get_model_data(self, card_feature_df):
    nt = get_non_text_feature_array(card_feature_df)
    if self.include_subtype_as_features:
      t = get_all_text_feature_array(card_feature_df)
    else:
      t = get_text_only_feature_array(card_feature_df)
    x = [np.concatenate((np.concatenate((nt[i,:], np.zeros((t[i].shape[1]-nt.shape[1])))).reshape((1,t[i].shape[1])), t[i]), axis=0) for i in range(nt.shape[0])]
    y = np.squeeze(get_price_category_array(card_feature_df)).tolist()
    x, y = self.clean_nan_data(x, y)
    return x, y

  # Train the LSTM to predict price category from text features.
  # Training must be done one sample at a time due to the variable input shape.
  def train(self, xtrain_list, ytrain_list, loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'], batch_size=256, epochs=5, verbose=2):
    if verbose:
      print("Training ShapelessTextModel...")
    self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics) # Previously used rmsprop optimizer
    # Model must be trained a single data point at a time to support variable length input sequences:
    state_reset = keras.callbacks.LambdaCallback(on_epoch_end=lambda *_ : self.model.reset_states())
    percent_complete, prev_percent_complete = 0, 0
    saved_model = self.model # Record the current model state
    for i in range(0, len(xtrain_list)):
      percent_complete = i/len(xtrain_list)*100
      # Report completion percentage
      if (percent_complete - prev_percent_complete)>3:
        prev_percent_complete = percent_complete
        print("\t", round(percent_complete,2), "% complete")
      xtrain = xtrain_list[i].reshape((1,xtrain_list[i].shape[0],xtrain_list[i].shape[1]))
      ytrain = np.array(ytrain_list[i]).reshape(1,1)
      self.model.fit(xtrain, ytrain, batch_size=1, epochs=epochs, verbose=0, shuffle=False, callbacks=[state_reset])
      # Check if the model is outputting nans. If so, restore the previous saved model
      test_output = self.predict(xtrain_list[0])
      if np.isnan(test_output).any():
        self.model = saved_model
        print("FOUND NAN -- reverting to previous model at index", i)
        print(test_output)
      else:
        saved_model = self.model
    self.trained = True

  # Predict output features (price category) for the given input features (text features).
  def predict(self, x, verbose=0):
    # If a single array of features is input, return a single prediction:
    if type(x)==np.ndarray:
      if len(x.shape)==2:
        x = x.reshape((1,x.shape[0],x.shape[1]))
      return self.model.predict(x, verbose=verbose)
    # If a list is input where each element is an array of features, return a list of predictions:
    elif type(x)==list:
      ypred = []
      for xi in x:
        if len(xi.shape)==2:
          xi = xi.reshape((1,xi.shape[0],xi.shape[1]))
        ypred.append(self.model.predict(xi, verbose=verbose))
      return ypred
    else:
      print("ERROR: the input x must be a numpy array or a list.")

  # Predict output features (price category) and evaluate model accuracy for the given input features (text features).
  def evaluate(self, x, y, verbose=0):
    ypred = self.predict(x)
    ypred_ints = convert_price_category_arrays_to_ints(ypred)
    loss = keras.losses.sparse_categorical_crossentropy(y, np.concatenate(ypred, axis=0))
    loss = loss.numpy().mean()
    accuracy = keras.metrics.Accuracy()
    accuracy.update_state(y, ypred_ints)
    accuracy = accuracy.result().numpy()
    precision, recall, fscore, support = classifier_metrics(y, ypred_ints)
    return {"sparse_categorical_crossentropy":loss, "accuracy":accuracy, "precision":precision, "recall":recall, "fscore":fscore, "support":support}

################################################################################
# NLP-Based Text Model With Fixed Length (Padded) Inputs 
# Uses an LSTM with all the textless/card statistics features of the StatsModel, but also 
# uses word vectors for each word in the card text and each subtype.
# Model inputs are padded to a single fixed input shape.
class PaddedTextModel:
  # Initialize PaddedTextModel by building an LSTM.
  # Choose whether to include nontext features along with rules text features by toggling include_nontext_as_features (default True)
  # Choose whether to include subtype features along with rules text features by toggling include_subtype_as_features (default False)
  def __init__(self, num_layers=2, num_units=256, dimension=50, sequence_length=130, verbose=False, nontext_condensed=True, include_nontext_as_context=True, include_nontext_as_features=False, include_subtype_as_features=False, loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy']):
    self.sequence_length=sequence_length
    if nontext_condensed:
      nontext_dim = len(non_text_feature_names_condensed())
    else:
      nontext_dim = len(non_text_feature_names())
    if include_nontext_as_context:
      input_dim = dimension + nontext_dim
    else:
      input_dim = dimension
    model = keras.Sequential()
    model.add(layers.LSTM(num_units, return_sequences=True, input_shape=(sequence_length, input_dim))) 
    for i in range(num_layers-1):
      model.add(layers.LSTM(num_units)) # specify only number of output units
    model.add(layers.Dense(len(list(price_categories())), activation="softmax"))
    if verbose:
      model.summary()
    self.model = model
    self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    self.nontext_condensed = nontext_condensed
    self.include_nontext_as_context = include_nontext_as_context
    self.include_nontext_as_features = include_nontext_as_features
    self.include_subtype_as_features = include_subtype_as_features
    self.loss=loss
    self.optimizer=optimizer
    self.metrics=metrics
    self.trained = False

  def __str__(self):
    return "PaddedTextModel"

  # Gets model data in the proper format for the PaddedTextModel.
  # Outputs:  x -- NxSxD numpy array where each SxD array is a single card's features
  #           y -- Nx1 numpy array where each element is a price category (int)
  def get_model_data(self, card_feature_df):
    if self.nontext_condensed:
      nt = get_condensed_non_text_feature_array(card_feature_df)
    else:
      nt = get_non_text_feature_array(card_feature_df)
    if self.include_subtype_as_features:
      t = get_all_text_feature_array(card_feature_df)
    else:
      t = get_text_only_feature_array(card_feature_df)
    # Combine non-text and text features:
    if self.include_nontext_as_features:
      x = [np.concatenate((np.concatenate((nt[i,:], np.zeros((t[i].shape[1]-nt.shape[1])))).reshape((1,t[i].shape[1])), t[i]), axis=0) for i in range(nt.shape[0])]
    else:
      if self.include_nontext_as_context:
        x = [np.concatenate((t[i], np.repeat(nt[i,:].reshape((1, nt[i,:].shape[0])), t[i].shape[0], axis=0)), axis=1) for i in range(nt.shape[0])]
      else:
        x = [t[i] for i in range(nt.shape[0])]
    # Trim extra sequence entries longer than the permitted sequence length:
    x = [xi[0:self.sequence_length, :] for xi in x]
    # Pad all sequences to the same sequence length:
    x = [np.concatenate((xi, np.zeros((self.sequence_length - xi.shape[0], xi.shape[1]))), axis=0) for xi in x]
    # Reshape the final output array:
    x = np.concatenate([np.expand_dims(xi, axis=0) for xi in x], axis=0)
    y = get_price_category_array(card_feature_df)
    return x, y

  # Train the LSTM to predict price category from text features.
  # Training must be done one sample at a time due to the variable input shape.
  def train(self, xtrain, ytrain, batch_size=256, epochs=25, verbose=2):
    if verbose:
      print("Training PaddedTextModel...")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="paddedtextmodel.ckpt", save_weights_only=True, verbose=1)
    self.model.fit(xtrain, ytrain, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=[cp_callback])
    self.trained = True

  # Predict output features (price category) for the given input features (text features).
  def predict(self, x, verbose=0):
    return self.model.predict(x, verbose=verbose)

  # Predict output features (price category) and evaluate model accuracy for the given input features (text features).
  def evaluate(self, x, y, verbose=0):
    ypred = self.predict(x)
    ypred_ints = convert_price_category_arrays_to_ints(ypred)
    loss = keras.losses.sparse_categorical_crossentropy(y, ypred)
    loss = loss.numpy().mean()
    accuracy = keras.metrics.Accuracy()
    accuracy.update_state(y, ypred_ints)
    accuracy = accuracy.result().numpy()
    precision, recall, fscore, support = classifier_metrics(y, ypred_ints)
    return {"sparse_categorical_crossentropy":loss, "accuracy":accuracy, "precision":precision, "recall":recall, "fscore":fscore, "support":support}

################################################################################
# NLP-Based Text Model With NonText And Text Features As Two Parallel Models
# Uses two individually trained models (one using only nontext features, the other using only text features).
# The outputs of each of those models are fed into a third and final predictive model
# Model inputs are padded to a single fixed input shape.
class ForkedTextModel:
  # Initialize ForkedTextModel with a PaddedTextModel and a StatsModel:
  # Choose whether to include nontext features along with rules text features by toggling include_nontext_as_features (default True)
  # Choose whether to include subtype features along with rules text features by toggling include_subtype_as_features (default False)
  def __init__(self, num_layers=2, num_units=256, dimension=50, sequence_length=130, verbose=False, nontext_condensed=True, include_nontext_as_context=False, include_nontext_as_features=False, include_subtype_as_features=False):
    self.paddedTextModel = PaddedTextModel(dimension=dimension, sequence_length=sequence_length, verbose=verbose, nontext_condensed=nontext_condensed, include_nontext_as_features=include_nontext_as_features, include_subtype_as_features=include_subtype_as_features)
    self.statsModel = StatsModel(condensed=nontext_condensed)
    model = keras.Sequential()
    model.add(layers.Input(shape=(2*len(list(price_categories())),)))
    for i in range(num_layers):
      model.add(layers.Dense(num_units, activation="relu"))
    model.add(layers.Dense(len(list(price_categories())), activation="softmax"))
    if verbose:
      model.summary()
    self.combinedModel = model
    self.trained = False

  def __str__(self):
    return "ForkedTextModel"

  # Gets model data in the proper format for the PaddedTextModel.
  # Outputs:  x -- NxSxD numpy array where each SxD array is a single card's features
  #           y -- Nx1 numpy array where each element is a price category (int)
  def get_model_data(self, card_feature_df):
    x_padded, y_padded = self.paddedTextModel.get_model_data(card_feature_df)
    x_stats, y_stats = self.statsModel.get_model_data(card_feature_df)
    return (x_padded, x_stats), (y_padded, y_stats)

  # Train both the PaddedTextModel and the StatsModel to predict price category from text features.
  # Training must be done one sample at a time due to the variable input shape.
  def train(self, xtrain_tuple, ytrain_tuple, loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'], batch_size=256, epochs=50, verbose=2):
    if verbose:
      print("Training ForkedTextModel...")
    x_padded, x_stats = xtrain_tuple
    y_padded, y_stats = ytrain_tuple
    self.paddedTextModel.train(x_padded, y_padded)
    self.statsModel.train(x_stats, y_stats)
    ypred_padded = self.paddedTextModel.predict(x_padded)
    ypred_stats = self.statsModel.predict(x_stats)
    x_combined, y_combined = np.concatenate([ypred_padded, ypred_stats], axis=1), y_padded
    self.combinedModel.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    self.combinedModel.fit(x_combined, y_combined, batch_size=batch_size, epochs=epochs, verbose=verbose)
    self.trained = True

  # Predict output features (price category) for the given input features (text features).
  def predict(self, x_tuple, verbose=0):
    x_padded, x_stats = x_tuple
    ypred_padded = self.paddedTextModel.predict(x_padded)
    ypred_stats = self.statsModel.predict(x_stats)
    return self.combinedModel.predict(np.concatenate([ypred_padded, ypred_stats], axis=1), verbose=verbose)

  # Predict output features (price category) and evaluate model accuracy for the given input features (text features).
  def evaluate(self, x_tuple, y_tuple, verbose=0):
    y, y_stats = y_tuple
    ypred = self.predict(x_tuple)
    ypred_ints = convert_price_category_arrays_to_ints(ypred)
    loss = keras.losses.sparse_categorical_crossentropy(y, ypred)
    loss = loss.numpy().mean()
    accuracy = keras.metrics.Accuracy()
    accuracy.update_state(y, ypred_ints)
    accuracy = accuracy.result().numpy()
    precision, recall, fscore, support = classifier_metrics(y, ypred_ints)
    return {"sparse_categorical_crossentropy":loss, "accuracy":accuracy, "precision":precision, "recall":recall, "fscore":fscore, "support":support}

##############################################################################################
# TESTING FUNCTIONS FOR CUSTOM AND EXISTING CARDS --------------------------------------------
##############################################################################################

# Create a custom card and obtain its feature dataframe.
def get_features_from_custom_card(name="Test", mana="{2}{u}", legendary=False, types="Sorcery", subtypes="", text="", power=-1, toughness=-1, loyalty=-1):
  st = "Legendary" if legendary else "-"
  card = {"Name":       [name],
          "ManaCost":   [mana],
          "Power":      [str(power)],
          "Toughness":  [str(toughness)],
          "Loyalty":    [str(loyalty)],
          "Text":       [text],
          "SuperTypes": [st],
          "CardTypes":  [types],
          "SubTypes":   [subtypes]}
  this_card = pd.DataFrame(card)
  return get_feature_df(this_card, initialize=False, verbose=False)

# Create a feature dataframe for an existing card with the input name.
def get_features_from_existing_card(name):
  try:
    return CARD_FEATURES[(CARD_FEATURES["Name"]==name) & (CARD_FEATURES["Permuted"]<2)]
  except:
    this_card = CARD_DF[CARD_DF["Name"]==name]
    return get_feature_df(this_card, initialize=False, verbose=False)

# For the input model, predict the price category for the input custom card.
def predict_custom_card(model, name="Test", mana="{2}{u}", legendary=False, types="Sorcery", subtypes="", text="", power=-1, toughness=-1, loyalty=-1):
  custom_df = get_features_from_custom_card(name=name, mana=mana, legendary=legendary, types=types, subtypes=subtypes, text=text, power=power, toughness=toughness, loyalty=loyalty)
  x, _ = model.get_model_data(custom_df)
  ypred = model.predict(x)
  print("Predicted output:", ypred)
  return ypred

# For the input model, predict the price category for the input card name. 
def predict_existing_card(model, cardname):
  df = get_features_from_existing_card(cardname)
  x, y = model.get_model_data(df)
  ypred = model.predict(x)
  if type(y)==tuple:
    y = y[0]
  print("Predicted output for "+cardname+": ", ypred, "  -->  ", convert_price_category_arrays_to_ints(ypred))
  print("True output for "+cardname+": ", y)
  return ypred, y


##############################################################################################
##############################################################################################
# TESTING ------------------------------------------------------------------------------------
##############################################################################################
##############################################################################################


reload_cards = False
reload_features = False
use_expanded_dataset = True
balance_dataset = True
retrain_model = False
test_set = "test"


# Load cards pandas dataframe:
if reload_cards: # Load cards from the card database. This takes several minutes.
    print("WARNING: Attempting to load all cards. This will take several minutes...")
    create_cards_csv() # Or create_cards_csv_by_set()
    CARD_DF = pd.read_csv('cards.csv')
else:
    CARD_DF = None
# Clean the card data into a dataframe that can be used to train a model:
CARD_DF = clean_card_data(CARD_DF) # CARD_DF.sample(n=10000, random_state=1)

# Get a dataframe with feature information for each card:
if reload_features:
    CARD_FEATURES = get_feature_df(CARD_DF, expanded=use_expanded_dataset)
    save_card_feature_df(CARD_FEATURES, filename="card_features.csv", expanded=use_expanded_dataset)
    save_scalers(expanded=use_expanded_dataset)
else:
    print("Loading card_features.csv...")
    CARD_FEATURES = load_card_feature_df(expanded=use_expanded_dataset)
    load_scalers(expanded=use_expanded_dataset)
print("Before balancing the dataset...")
print(CARD_FEATURES.groupby(["PriceCategory"]).count()["Name"])

# Retrain the model and evaluate its performance, then save the trained model.
if retrain_model:
    # models = [FrequencyModel(), StatsModel(condensed=False), StatsModel(condensed=True), PaddedTextModel()] # ShapelessTextModel(), ForkedTextModel()
    # models = [PaddedTextModel(include_nontext_as_context=True)]
    models = [FrequencyModel(), StatsModel(condensed=False)]
    CARDS_TRAIN, cvalid, ctest = split_dataframe(CARD_FEATURES)
    CARDS_VALID, CARDS_VALID_PERM = cvalid
    CARDS_TEST,  CARDS_TEST_PERM  = ctest
    for model in models:
      if test_set == "test":
        try:
            CARDS_TRAIN_THIS_MODEL = pd.concat([CARDS_TRAIN, CARDS_VALID, CARDS_VALID_PERM], ignore_index=True).reset_index(drop=True)
        except:
            CARDS_TRAIN_THIS_MODEL = pd.concat([CARDS_TRAIN, CARDS_VALID], ignore_index=True).reset_index(drop=True)
      else:
        CARDS_TRAIN_THIS_MODEL = CARDS_TRAIN
      if type(model)!=FrequencyModel and balance_dataset:
        CARDS_TRAIN_THIS_MODEL = balance_training_dataset(CARDS_TRAIN_THIS_MODEL)
        print("After balancing the dataset...")
        print(CARDS_TRAIN_THIS_MODEL.groupby(["PriceCategory"]).count()["Name"])
      xtrain, ytrain = model.get_model_data(CARDS_TRAIN_THIS_MODEL)
      xvalid, yvalid = model.get_model_data(CARDS_VALID)
      xtest,  ytest =  model.get_model_data(CARDS_TEST)
      model.train(xtrain, ytrain)
      if test_set == "test":
        evals = model.evaluate(xtest, ytest)
      else:
        evals = model.evaluate(xvalid, yvalid)
      print("\nEVAL RESULTS...")
      [print(k, ": ", v) for (k,v) in zip(evals.keys(), evals.values())]
      print("\n")

# Experiment on the previously trained model.
else:
    model = PaddedTextModel(include_nontext_as_context=True)
    model.model.load_weights("paddedtextmodel.ckpt")
    
    existing_cards = ["Vorinclex, Monstrous Raider", "Colossal Dreadmaw", "Bloom Tender", "Ancestor's Chosen", "Angel of Mercy", "Angelic Blessing", "Skullclamp", "Sword of the Animist", "Draugr's Helm", "Manriki-Gusari", "Batterbone", "Sword of Hours", "Swiftfoot Boots", "Lightning Greaves", "Tenza, Godo's Maul", "Blasphemous Act", "Damn", "Mystic Remora", "Vandalblast", "Eerie Ultimatum"]
    for card in existing_cards:
        predict_existing_card(model, card)
    print("\n\n\n")

    predict_custom_card(model, types="Enchantment", mana="{1}{u}{u}", text="Whenever a creature you control attacks, draw a card.\nWhenever a creature dies, mill three cards.")
    predict_custom_card(model, types="Enchantment", mana="{1}{u}{u}", text="Whenever a creature you control attacks, draw a card.\nWhenever a creature dies, draw three cards.")
    predict_custom_card(model, types="Enchantment", mana="{4}{u}", text="When this permanent enters the battlefield, draw a card.")
    predict_custom_card(model, types="Enchantment", mana="{4}{u}", text="When this permanent enters the battlefield, mill four cards, then return two cards from your graveyard to your hand.")
    predict_custom_card(model, types="Instant", mana="{u}", text="Counter target spell.")
    predict_custom_card(model, types="Instant", mana="{u}{u}", text="Counter target noncreature spell.")
    predict_custom_card(model, types="Instant", mana="{2}{u}", text="Counter target spell. Its controller draws a card.")
    predict_custom_card(model, types="Instant", mana="{w}", text="Exile target creature.")
    predict_custom_card(model, types="Instant", mana="{2}{w}", text="Exile target creature. Its controller draws a card.")


    print("\n\n\n")
