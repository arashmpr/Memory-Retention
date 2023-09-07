class Statistics():
  def __init__(self, filename, glove_model):
    self.df = pd.read_csv(filename)
    self.path = "/content/gdrive/MyDrive/Memory Retention Research/Datasets/Dataset_10000/Dataset_10000_temp.csv"

    null_rows = self.df.isnull().any(axis=1)
    rows_with_null_data = self.df[null_rows]
    paragraph = rows_with_null_data['Paragraph']
    self.df = self.df[~self.df['Paragraph'].isin(paragraph)]
    self.glove_model = glove_model

    with open('bias_lexicon.txt', 'r') as file:
      self.bias_words = file.read().splitlines()

    self.vad_df = pd.read_csv('NRC-VAD-Lexicon.txt', delimiter='\t', names=['Words', 'valence', 'arousal', 'dominance'])
    self.vad_words = set(self.vad_df['Words'].values)

  def calculate_C(self):
    C = []
    paragraphs = self.df['Paragraph'].values
    for paragraph in paragraphs:
      tokens = paragraph.split()
      C.append(len(tokens))
    return C

  def calculate_d(self):
    d = []
    paragraphs = self.df['Paragraph'].values
    target_words = self.df['Target Word'].values
    for i, paragraph in enumerate(paragraphs):
      tokens = paragraph.split()
      length = len(tokens)
      if target_words[i] in tokens:
        ind = tokens.index(target_words[i])
        diff = length - ind - 1
      else:
        diff = -1
      d.append(diff)
    return d

  def calculate_most_similar_and_similarity(self):
    sim_words = []
    sim_scores = []
    target_words = self.df['Target Word'].values
    for target_word in tqdm(target_words):
      if target_word in self.glove_model.key_to_index:
        sim_word, sim_score = self.glove_model.most_similar(target_word)[0]
        sim_words.append(sim_word)
        sim_scores.append(sim_score)
      else:
        sim_words.append(-1)
        sim_scores.append(-1)

    assert len(target_words) == len(sim_words)

    return sim_words, sim_scores

  def calculate_repetition(self):
    repetition = []
    paragraphs = self.df['Paragraph'].values
    target_words = self.df['Target Word'].values

    for i, paragraph in enumerate(paragraphs):
      repeat = paragraph.split().count(target_words[i])
      repetition.append(repeat)
    return repetition

  def is_vad(self):
    is_vad = []

    target_words = self.df['Target Word'].values
    for target_word in target_words:
      if target_word in self.vad_words:
        is_vad.append(1)
      else:
        is_vad.append(0)

    return is_vad

  def is_bias(self):
    is_bias = []

    target_words = self.df['Target Word'].values
    for target_word in target_words:
      if target_word in self.bias_words:
        is_bias.append(1)
      else:
        is_bias.append(0)

    return is_bias

  def calculate_POS(self):
    poses = []

    paragraphs = self.df[self.df['Label']==1]['Paragraph'].values
    target_words = self.df[self.df['Label']==1]['Target Word'].values

    for i, paragraph in tqdm(enumerate(paragraphs)):
      words = nltk.word_tokenize(paragraph)
      pos_tags = nltk.pos_tag(words)

      for word, pos in pos_tags:
        if word == target_words[i]:
          poses.append(pos)
          break

    neg_poses = [-1 for _ in range(len(poses))]
    poses.extend(neg_poses)
    return poses

  def calculate_unigram_frequency(self):
    word_freq = []

    paragraphs = np.unique(self.df['Paragraph'].values)
    paragraphs = list(paragraphs)
    document = ' '.join(paragraphs)

    words = document.split()
    word_frequency = Counter(words)

    target_words = self.df['Target Word'].values
    for word in target_words:
      if word in word_frequency:
        word_freq.append(word_frequency[word])
      else:
        word_freq.append(0)
    return word_freq

  def calculate_query_length(self):
    query_length = []
    words = self.df['Target Word'].values
    for query in words:
      query_length.append(len(query))
    return query_length

  def save(self, filename, base_path):
    self.df.to_csv(filename, index=False)

    pos_df = self.df[self.df['Label']==1]
    neg_df = self.df[self.df['Label']==0]

    pos_train_df, pos_remaining = train_test_split(pos_df, test_size=0.1, random_state=42)
    pos_valid_df, pos_test_df = train_test_split(pos_remaining, test_size=0.5, random_state=42)

    neg_train_df, neg_remaining = train_test_split(neg_df, test_size=0.1, random_state=42)
    neg_valid_df, neg_test_df = train_test_split(neg_remaining, test_size=0.5, random_state=42)

    train_df = pd.concat([pos_train_df, neg_train_df])
    test_df = pd.concat([pos_test_df, neg_test_df])
    val_df = pd.concat([pos_valid_df, neg_valid_df])

    train_df.to_csv( base_path + "Dataset_10000/train.csv", index=False)
    test_df.to_csv(base_path + "Dataset_10000/test.csv", index=False)
    val_df.to_csv(base_path + "Dataset_10000/val.csv", index=False)

  def save_periodically(self):
    self.df.to_csv(self.path, index=False)

  def run_statistics(self):
    self.df['C'] = self.calculate_C()
    self.save_periodically()
    print("C is calculated and saved")

    self.df['d'] = self.calculate_d()
    self.save_periodically()
    print("d is calculated and saved")

    self.df['most_similar'], self.df['similarity_score'] = self.calculate_most_similar_and_similarity()
    self.save_periodically()
    print("most similar and similarity is calculated and saved")

    self.df['POS'] = self.calculate_POS()
    self.save_periodically()
    print("POS is calculated and saved")

    self.df['repetitions'] = self.calculate_repetition()
    self.save_periodically()
    print("repetitions is calculated and saved")

    self.df['is_vad'] = self.is_vad()
    self.save_periodically()
    print("vad is calculated and saved")

    self.df['is_bias'] = self.is_bias()
    self.save_periodically()
    print("bias is calculated and saved")

    self.df['unigram_frequency'] = self.calculate_unigram_frequency()
    self.save_periodically()
    print("unigram_frequency is calculated and saved")

    self.df['query_length'] = self.calculate_query_length()
    self.save_periodically()
    print("length of query is calculated and saved")

print(dataset_path)

statistics = Statistics(dataset_path, glove_model)

statistics.run_statistics()

statistics.save(dataset_path, base_path)