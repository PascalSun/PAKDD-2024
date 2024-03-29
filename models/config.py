# node2vec train params
NODE2VEC_DIM = 50
NODE2VEC_WALK_LENGTH = 30
NODE2VEC_NUM_WALKS = 10
NODE2VEC_WORKER = 4

NODE2VEC_WINDOW = 20  # TODO: reduce it down
NODE2VEC_MIN_COUNT = 1
NODE2VEC_BATCH_WORD = 128
NODE2VEC_P = 1
NODE2VEC_Q = 1

RISK_LABELS = ["Fatal", "Hospital", "Medical", "PDO Major", "PDO Minor", "None"]
RISK_LABELS.reverse()
