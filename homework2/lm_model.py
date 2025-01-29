from collections import Counter
import random

"""
Kate Lanman
CS 4120, Spring 2025
Homework 2
"""

# constants
SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"
UNK = "<UNK>"


# UTILITY FUNCTIONS

def create_ngrams(tokens: list, n: int, sentence_end = SENTENCE_END) -> list:
  """Creates n-grams for the given token sequence.
  Args:
    tokens (list): a list of tokens as strings
    n (int): the length of n-grams to create

  Returns:
    list: list of tuples of strings, each tuple being one of the individual n-grams
  """
  if n == 1:   # unigrams
    return tokens
  
  if len(tokens) <= n:
    return [tuple(tokens)]

  return [tuple(tokens[i: i+n]) for i in range(len(tokens)-n+1) if tokens[i] != sentence_end]

def read_file(path: str) -> list:
  """
  Reads the contents of a file in line by line.
  Args:
    path (str): the location of the file to read

  Returns:
    list: list of strings, the contents of the file
  """
  # PROVIDED
  f = open(path, "r", encoding="utf-8")
  contents = f.readlines()
  f.close()
  return contents

def tokenize_line(line: str, ngram: int, 
                   by_char: bool = True, 
                   sentence_begin: str=SENTENCE_BEGIN, 
                   sentence_end: str=SENTENCE_END):
  """
  Tokenize a single string. Glue on the appropriate number of 
  sentence begin tokens and sentence end tokens (ngram - 1), except
  for the case when ngram == 1, when there will be one sentence begin
  and one sentence end token.
  Args:
    line (str): text to tokenize
    ngram (int): ngram preparation number
    by_char (bool): default value True, if True, tokenize by character, if
      False, tokenize by whitespace
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

  Returns:
    list of strings - a single line tokenized
  """
  # PROVIDED
  inner_pieces = None
  if by_char:
    inner_pieces = list(line)
  else:
    # otherwise split on white space
    inner_pieces = line.split()

  if ngram == 1:
    tokens = [sentence_begin] + inner_pieces + [sentence_end]
  else:
    tokens = ([sentence_begin] * (ngram - 1)) + inner_pieces + ([sentence_end] * (ngram - 1))
  # always count the unigrams
  return tokens


def tokenize(data: list, ngram: int, 
                   by_char: bool = True, 
                   sentence_begin: str=SENTENCE_BEGIN, 
                   sentence_end: str=SENTENCE_END):
  """
  Tokenize each line in a list of strings. Glue on the appropriate number of 
  sentence begin tokens and sentence end tokens (ngram - 1), except
  for the case when ngram == 1, when there will be one sentence begin
  and one sentence end token.
  Args:
    data (list): list of strings to tokenize
    ngram (int): ngram preparation number
    by_char (bool): default value True, if True, tokenize by character, if
      False, tokenize by whitespace
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

  Returns:
    list of strings - all lines tokenized as one large list
  """
  # PROVIDED
  total = []
  # also glue on sentence begin and end items
  for line in data:
    line = line.strip()
    # skip empty lines
    if len(line) == 0:
      continue
    tokens = tokenize_line(line, ngram, by_char, sentence_begin, sentence_end)
    total += tokens
  return total


class LanguageModel:

  def __init__(self, n_gram, sentence_begin=SENTENCE_BEGIN, sentence_end=SENTENCE_END, unk=UNK):
    """Initializes an untrained LanguageModel
    Args:
      n_gram (int): the n-gram order of the language model to create
    """
    self.n = n_gram

    # special tokens
    self.sentence_begin = sentence_begin
    self.sentence_end = sentence_end
    self.unk = unk
    
    self.total = 0
    self.vocabulary = ()
    self.n_grams = {}

    # n - 1 grams for non unigram probability calculations
    self.n_less = {}
  
  def train(self, tokens: list, verbose: bool = False) -> None:
    """Trains the language model on the given data. 
    Args:
      tokens (list): tokenized data to be trained on as a single list
      verbose (bool): default value False, to be used to turn on/off debugging prints
    """
    self.total = len(tokens)

    counts = Counter(tokens)
    if verbose: print(f'counts: {counts}')

    # tokenize and store if no UNK mapping needed
    if 1 not in counts.values():
      self.vocabulary = set(tokens)
      self.n_grams = Counter(create_ngrams(tokens, self.n))
      self.n_less = Counter(create_ngrams(tokens, self.n - 1)) if self.n > 1 else None

      if verbose: print('no tokens with count 1 found')
      return
    
    # map tokens to UNK if token occurs only once in the training set
    tokens = [w if counts[w] > 1 else self.unk for w in tokens]

    if verbose: print(f'tokens with <UNK>: {tokens}')

    self.vocabulary = set(tokens)
    self.n_grams = Counter(create_ngrams(tokens, self.n))
    self.n_less = Counter(create_ngrams(tokens, self.n - 1)) if self.n > 1 else None

    if verbose:
      print(f'ngrams: {self.n_grams.most_common(5)}')
      print(f'n - 1 grams: {self.n_less.most_common(5)}')

  def ngram_p(self, ngram: list) -> float:
    """Calculates the probability of a given ngram.
        P(w_i|w_{i-n+1}...w_{i-1})
    Args:
      ngram (list): ngram to get probability score for
    
    Returns:
      float: the probability value of the ngram
    """
    # check ngram is proper size
    if isinstance(ngram, str) or self.n == 1:
      if self.n != 1 or not isinstance(ngram, str):
        raise ValueError(f"Expected ngram of length {self.n}, got {ngram}")
    elif len(ngram) != self.n:
      raise ValueError(f"Expected ngram of length {self.n}, got {len(ngram)}{ngram}")

    # unigram probability
    if self.n == 1:
      return self.n_grams[ngram] / self.total
    
    # get first n - 1 tokens
    prev = ngram[:-1]
    prev = tuple(prev) if self.n - 1 > 1 else prev[0] # prev as single string if n is 2

    return self.n_grams[tuple(ngram)] / self.n_less[prev]

  def laplace_p(self, ngram: list) -> float:
    """Calculates the laplace smoothed probability of a given ngram.
        P(w_i|w_{i-n+1}...w_{i-1})
    Args:
      ngram (list): ngram to get probability score for
    
    Returns:
      float: the probability value of the ngram
    """
    # check ngram is proper size
    if isinstance(ngram, str) or self.n == 1:
      if self.n != 1 or not isinstance(ngram, str):
        raise ValueError(f"Expected ngram of length {self.n}, got {ngram}")
    elif len(ngram) != self.n:
      raise ValueError(f"Expected ngram of length {self.n}, got {len(ngram)}{ngram}")

    # unigram probability
    if self.n == 1:
      return (self.n_grams[ngram] + 1) / (self.total + len(self.vocabulary))
    
    # get first n - 1 tokens
    prev = ngram[:-1]
    prev = tuple(prev) if self.n - 1 > 1 else prev[0] # prev as single string if n is 2

    return (self.n_grams[tuple(ngram)] + 1) / (self.n_less[prev] + len(self.vocabulary))

  def score(self, sentence_tokens: list) -> float:
    """Calculates the probability score for a given list representing a single sequence of tokens.
    Args:
      sentence_tokens (list): a tokenized sequence to be scored by this model
      
    Returns:
      float: the probability value of the given tokens for this model
    """
    # replace unknown tokens
    sentence_tokens = [x if x in self.vocabulary else self.unk for x in sentence_tokens]

    P = 1

    # get probability for each P(wi | w{i-n+1}...w{i-1})
    n_tokens = len(sentence_tokens)
    for i in range(n_tokens - self.n + 1):
      if self.n == 1:
        ngram = sentence_tokens[i] # unigrams
      elif n_tokens < self.n:
        ngram = ([self.sentence_begin] * (self.n - n_tokens)) + [sentence_tokens[i]] # add start token to make ngram
      else:
        ngram = sentence_tokens[i: i + self.n]

      p = self.laplace_p(ngram)
      P = P * p

    return P

  def generate_sentence(self) -> list:
    """Generates a single sentence from a trained language model using the Shannon technique,
    returning it as a list of tokens.
      
    Returns:
      list: the generated sentence as a list of tokens
    """
    # initialize sentence with <s>
    sentence = [self.sentence_begin] * max(1, self.n - 1)
    next_token = ""

    while next_token != self.sentence_end:

      # get probabilities for each word coming next
      if self.n == 1: # unigrams
        probs = {w: self.ngram_p(w) for w in self.vocabulary if w != '<s>'}
      else: # n > 1 grams
        probs = {w: self.ngram_p(sentence[-(self.n - 1):] + [w]) for w in self.vocabulary if w != '<s>'}

      # sample next token based on weights
      next_token = random.choices(list(probs.keys()), weights=list(probs.values()), k=1)[0]
      sentence.append(next_token)
    
    return sentence
        

  def generate(self, n: int) -> list:
    """Generates n sentences from a trained language model using the Shannon technique.
    Args:
      n (int): the number of sentences to generate
      
    Returns:
      list: a list containing lists of strings, one per generated sentence
    """
    # PROVIDED
    return [self.generate_sentence() for i in range(n)]

if __name__ == '__main__':
  pass