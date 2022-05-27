import omniglot


BATCH_SIZE = 16
NUM_WAY = 5
NUM_SUPPORT = 1
NUM_QUERY = 19
NUM_TRAINING_ITERATIONS = 15000
NUM_TRAINING_TASKS = BATCH_SIZE*(NUM_TRAINING_ITERATIONS)
NUM_TEST_TASKS = 600

trainloader = omniglot.get_omniglot_dataloader(
    'train',
    BATCH_SIZE,
    NUM_WAY,
    NUM_SUPPORT,
    NUM_QUERY,
    NUM_TRAINING_TASKS
)

validloader = omniglot.get_omniglot_dataloader(
    'val',
    BATCH_SIZE,
    NUM_WAY,
    NUM_SUPPORT,
    NUM_QUERY,
    BATCH_SIZE * 4
)

testloader = omniglot.get_omniglot_dataloader(
    'test',
    1,
    NUM_WAY,
    NUM_SUPPORT,
    NUM_QUERY,
    NUM_TEST_TASKS
)
