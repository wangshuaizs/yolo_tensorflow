from __future__ import print_function

import os
import re
import argparse
import datetime
import tensorflow as tf
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer
from utils.pascal_voc import pascal_voc

slim = tf.contrib.slim


class Solver(object):

    def __init__(self, net, data, params, num_workers, server):
        self.net = net
        self.data = data
        self.weights_file = cfg.WEIGHTS_FILE
        self.max_iter = cfg.MAX_ITER
        self.initial_learning_rate = cfg.LEARNING_RATE
        self.decay_steps = cfg.DECAY_STEPS
        self.decay_rate = cfg.DECAY_RATE
        self.staircase = cfg.STAIRCASE
        self.summary_iter = cfg.SUMMARY_ITER
        self.save_iter = cfg.SAVE_ITER
        self.output_dir = os.path.join(
            cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.save_cfg()

        self.variable_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        self.ckpt_file = os.path.join(self.output_dir, 'yolo')
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)

        self.global_step = tf.train.create_global_step()
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, self.staircase, name='learning_rate')
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate)
        if params.job_name is not None:
            self.optimizer = tf.train.SyncReplicasOptimizer(self.optimizer, 
                replicas_to_aggregate=num_workers, 
                total_num_replicas=num_workers)
        self.train_op = slim.learning.create_train_op(
            self.net.total_loss, self.optimizer, global_step=self.global_step)

        self.is_chief = 0
        if params.job_name == "worker" and params.task_index == 0:
            self.is_chief = 1
            chief_queue_runner = self.optimizer.get_chief_queue_runner()
            init_tokens_op = self.optimizer.get_init_tokens_op()

        self.init_op = tf.global_variables_initializer()
        self.sv = tf.train.Supervisor(is_chief=self.is_chief,
                                 init_op=self.init_op,
                                 recovery_wait_secs=1,
                                 global_step=self.global_step)

        gpu_options = tf.GPUOptions()
        config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=cfg.LOG_DEVICE_PLACEMENT)
        if self.is_chief:
            print("Worker %d: Initializing session..." % params.task_index)
        else:
            print("Worker %d: Waiting for session to be initialized..." % params.task_index)
        self.sess = self.sv.prepare_or_wait_for_session(server.target, config=config)
        print("Worker %d: Session initialization complete." % params.task_index)
        if self.is_chief:
            print("Starting chief queue runner and running init_tokens_op")
            self.sv.start_queue_runners(self.sess, [chief_queue_runner])
            self.sess.run(init_tokens_op)

        if self.weights_file is not None:
            print('Restoring weights from: ' + self.weights_file)
            self.saver.restore(self.sess, self.weights_file)

        self.writer.add_graph(self.sess.graph)

    def train(self):

        train_timer = Timer()
        load_timer = Timer()

        for step in range(1, self.max_iter + 1):

            load_timer.tic()
            images, labels = self.data.get()
            load_timer.toc()
            feed_dict = {self.net.images: images,
                         self.net.labels: labels}

            if step % self.summary_iter == 0:
                if step % (self.summary_iter * 10) == 0:

                    train_timer.tic()
                    summary_str, loss, _ = self.sess.run(
                        [self.summary_op, self.net.total_loss, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()

                    log_str = "{} Epoch: {}, Step: {}, Learning rate: {}, Loss: {:5.3f}\nSpeed: {:.3f}s/iter, Load: {:.3f}s/iter, Remain: {}".format(
                        datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                        self.data.epoch,
                        int(step),
                        round(self.learning_rate.eval(session=self.sess), 6),
                        loss,
                        train_timer.average_time,
                        load_timer.average_time,
                        train_timer.remain(step, self.max_iter))
                    print(log_str)

                else:
                    train_timer.tic()
                    summary_str, _ = self.sess.run(
                        [self.summary_op, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()

                self.writer.add_summary(summary_str, step)

            else:
                train_timer.tic()
                self.sess.run(self.train_op, feed_dict=feed_dict)
                train_timer.toc()

            if step % self.save_iter == 0:
                print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    self.output_dir))
                self.saver.save(
                    self.sess, self.ckpt_file, global_step=self.global_step)

    def save_cfg(self):

        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)


def update_config_paths(data_dir, weights_file):
    cfg.DATA_PATH = data_dir
    cfg.PASCAL_PATH = os.path.join(data_dir, 'pascal_voc')
    cfg.CACHE_PATH = os.path.join(cfg.PASCAL_PATH, 'cache')
    cfg.OUTPUT_DIR = os.path.join(cfg.PASCAL_PATH, 'output')
    cfg.WEIGHTS_DIR = os.path.join(cfg.PASCAL_PATH, 'weights')

    cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, weights_file)

def print_parameter_info():
    parameter_count = 0
    print("="*64)
    for var in tf.trainable_variables():
        print(var.name, end='\t')
        print("@%s" % var.device, end='\t')
        print(var.shape)
        parameter_count = parameter_count + reduce(lambda x, y: x * y, var.get_shape().as_list())
    print("Total parameter : %d, i.e., %.0f MB" % (parameter_count, parameter_count/1024/256))
    print("="*64)  

def distributed_setup(params):
    ps_hosts = re.findall(r'[\w\.:]+', params.ps_hosts)
    num_parameter_servers = len(ps_hosts)
    worker_hosts = re.findall(r'[\w\.:]+', params.worker_hosts)
    num_workers = len(worker_hosts)
    server = tf.train.Server({"ps":ps_hosts,"worker":worker_hosts}, 
        job_name=params.job_name, 
        task_index=params.task_index, 
        protocol=params.server_protocol)

    if params.job_name == "ps":
        server.join()
    else:
        is_chief = (params.task_index == 0)

    cluster_spec = tf.train.ClusterSpec({"ps":ps_hosts,"worker":worker_hosts})

    return is_chief, tf.train.replica_device_setter(cluster=cluster_spec), num_workers, server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--threshold', default=0.2, type=float)
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--gpu', default='', type=str)
    parser.add_argument('--log_device_placement', default='', type=bool)
    parser.add_argument('--batch_size', default='', type=int)
    parser.add_argument('--ps_hosts', default='', type=str)
    parser.add_argument('--worker_hosts', default='', type=str)
    parser.add_argument('--job_name', default='', type=str)
    parser.add_argument('--task_index', default='', type=int)
    parser.add_argument('--server_protocol', default='', type=str)
    args = parser.parse_args()

    if args.gpu is not None:
        cfg.GPU = args.gpu
    if args.log_device_placement is not None:
        cfg.LOG_DEVICE_PLACEMENT = args.log_device_placement
    if args.batch_size is not None:
        cfg.BATCH_SIZE = args.batch_size

    if args.data_dir != cfg.DATA_PATH:
        update_config_paths(args.data_dir, args.weights)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    is_chief, device_setter, num_workers, server = distributed_setup(args)
    yolo = YOLONet(device_setter=device_setter)
    pascal = pascal_voc('train')

    solver = Solver(yolo, pascal, args, num_workers, server)

    print_parameter_info()

    print('Start training ...')
    solver.train()
    print('Done training.')


if __name__ == '__main__':

    # python train.py --weights YOLO_small.ckpt --gpu 0
    main()
