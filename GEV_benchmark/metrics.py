def correct_instance(y, y_hat, l_len, return_correct_list=False):
    b = y.shape[0]
    correct_count = 0
    correct_list = []
    for i in range(b):
        y_ = y[i][0: l_len[i]]
        y_hat_ = y_hat[i][0: l_len[i]]
        y_ = y_ >= 0.5
        y_hat_ = y_hat_ >= 0.5
        if (y_ == y_hat_).all():
            correct_count += 1
            correct_list.append(1)
        else:
            correct_list.append(0)
    if return_correct_list:
        return correct_count, correct_list
    return correct_count