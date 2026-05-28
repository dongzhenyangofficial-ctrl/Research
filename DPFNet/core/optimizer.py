from paddle.optimizer import Adam


def get_optim(opt, parameters):
    # print(len([_ for _ in filter(lambda p: p.trainable, parameters)]), len([_ for _ in filter(lambda p: not p.trainable, parameters)]), len([_ for _ in parameters]))
    optimizer = Adam(parameters=filter(lambda p: p.trainable, parameters),
                     learning_rate=opt.learning_rate,
                     weight_decay=opt.weight_decay)
    return optimizer
