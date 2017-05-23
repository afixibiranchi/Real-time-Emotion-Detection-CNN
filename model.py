import tensorflow as tf

class Model:
    
    def __init__(self, dimension_picture, learning_rate, dimension_output, enable_dropout, dropout, enable_penalty, penalty):
        
        self.X = tf.placeholder(tf.float32, (None, dimension_picture, dimension_picture, 1))
        self.Y = tf.placeholder(tf.float32, (None, dimension_output))
        
        def convolutionize(x, w):
            return tf.nn.conv2d(input = x, filter = w, strides = [1, 1, 1, 1], padding = 'SAME')
        
        def pooling(wx):
            return tf.nn.max_pool(wx, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        
        first_W_conv = tf.Variable(tf.random_normal([5, 5, 1, 4], stddev = 0.5))
        first_b_conv = tf.Variable(tf.random_normal([4], stddev = 0.1))
        first_hidden_conv = tf.nn.relu(convolutionize(self.X, first_W_conv) + first_b_conv)
        first_hidden_pool = pooling(first_hidden_conv)
        
        if enable_dropout:
            first_hidden_pool = tf.nn.dropout(first_hidden_pool, dropout / 10.0)
        
        second_W_conv = tf.Variable(tf.random_normal([5, 5, 4, 8], stddev = 0.5))
        second_b_conv = tf.Variable(tf.random_normal([8], stddev = 0.1))
        second_hidden_conv = tf.nn.relu(convolutionize(first_hidden_pool, second_W_conv) + second_b_conv)
        second_hidden_pool = pooling(second_hidden_conv)
        
        if enable_dropout:
            second_hidden_pool = tf.nn.dropout(second_hidden_pool, dropout / 5.0)
            
        third_W_conv = tf.Variable(tf.random_normal([5, 5, 8, 16], stddev = 0.5))
        third_b_conv = tf.Variable(tf.random_normal([16], stddev = 0.1))
        third_hidden_conv = tf.nn.sigmoid(convolutionize(second_hidden_pool, third_W_conv) + third_b_conv)
        third_hidden_pool = pooling(third_hidden_conv)
        
        if enable_dropout:
            third_hidden_pool = tf.nn.dropout(third_hidden_pool, dropout / 3.0)
        
        fourth_W_conv = tf.Variable(tf.random_normal([2304, 128], stddev = 0.5))
        fourth_b_conv = tf.Variable(tf.random_normal([128], stddev = 0.1))
        fourth_hidden_flatted = tf.reshape(third_hidden_pool, [-1, 2304])
        fourth_hidden_conv = tf.nn.sigmoid(tf.matmul(fourth_hidden_flatted, fourth_W_conv) + fourth_b_conv)
        
        if enable_dropout:
            fourth_hidden_conv = tf.nn.dropout(fourth_hidden_conv, dropout / 2.0)
        
        W = tf.Variable(tf.random_normal([128, dimension_output], stddev = 0.5))
        b = tf.Variable(tf.random_normal([dimension_output], stddev = 0.1))
        
        self.y_hat = tf.matmul(fourth_hidden_conv, W) + b
        
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y_hat, self.Y))
        
        if enable_penalty:
            regularizers = tf.nn.l2_loss(first_W_conv) + tf.nn.l2_loss(second_W_conv) + tf.nn.l2_loss(third_W_conv) + tf.nn.l2_loss(fourth_W_conv)
            self.cost = tf.reduce_mean(self.cost + penalty * regularizers)
        
        self.optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(self.cost)
        
        correct_prediction = tf.equal(tf.argmax(self.y_hat, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        