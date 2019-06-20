from multiprocessing import Pool as ProcessPool


def iterate_list_with_parallel(f, data_length, thread):
    p = ProcessPool(thread)  # Ten Threads
    index_list = [(int(idx * data_length / thread), int((idx+1) * data_length/thread))  for idx in range(thread)]
    p.map(f, index_list)
    p.close()
    p.join()


if __name__ == "__main__":
    mapping = [i for i in range(1000)]

    def foo(idx):

        """
        index_list is just a idx that we transferred into this foo
        :param index_list:
        :return:
        """

        start_idx, end_idx = idx

        for each in mapping[start_idx: end_idx]:
            print(each)

    iterate_list_with_parallel(foo, 1000, 10)