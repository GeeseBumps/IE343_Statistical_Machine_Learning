
#K Fold cross validation 사용해보자! Ridge regularization 사용
        accuracy_list = []
        w_list = []
        data_len= int(len(X)/5)

        for j in range(5):
            validation_len = slice(j * data_len, (j + 1) * data_len)
            if j==4:
                validation_len = slice(j*data_len,len(X)+1)

            train_X = np.delete(X,validation_len,axis=0)
            train_y= np.delete(y,validation_len)
            validation_X = X[validation_len,:]
            validation_y = y[validation_len]


            ridge_lambda=np.exp(-4)
            for _ in range(epoch_num):
                w_prev = np.copy(w)
                y_hat = 1/(1+np.exp(-train_X @ w))
                grad = np.matmul((y_hat - train_y), train_X)+(ridge_lambda)*w_prev
                w -= lr*grad
                if np.allclose(w, w_prev):
                    break
            pred_y = self.predict(validation_X)
            pred_y[pred_y < 0.5] = 0
            pred_y[pred_y >= 0.5] = 1
            accuracy = np.sum(validation_y == pred_y) / len(validation_y) * 100
            accuracy_list.append(accuracy)
            w_list.append(w)
        print('Accuracy list in k-fold cross validation :',accuracy_list)
        w = w_list[np.argmax(accuracy_list)]
        print(w)
