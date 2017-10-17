#!/usr/bin/env python3
import
import numpy as np

def predict(weekday, model_dict, model):
    gen_hrs = list(range(0, 24))

    # Weekday
    get_w_lbl = model_dict['lbl_w'].transform([weekday])
    get_w_ohe = model_dict['ohe_w'].transform([get_w_lbl]).toarray()

    mini_batch = []
    # Hour
    for i in gen_hrs:
        get_h_ohe = model_dict['ohe_h'].transform(i).toarray()
        all_arr = np.hstack((get_w_ohe, get_h_ohe))
        mini_batch.append(all_arr)

    mini_batch = np.array(mini_batch).reshape(-1, 31)
    yhats_train = model.predict(mini_batch, batch_size=24)
    max_y_value = np.argmax(yhats_train, axis=1)
    return max_y_value



if __name__ == '__main__':
    wd = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for i in wd:
        predict(i)
