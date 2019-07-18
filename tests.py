import unittest

from map_reduce import Task, Worker, MapReduce

mapper = lambda s: (s, 1)
reducer = lambda x, y: x + y


class TestClassifierWrapper(unittest.TestCase):
    def test_task(self):
        task = Task('peter', [1, 1, 1, 1], reducer)
        self.assertEqual(('peter', 4), task.run())

    def test_worker(self):
        worker = Worker()
        task_1 = Task('peter', [1, 1, 1, 1], reducer)
        task_2 = Task('jenny', [1, 1], reducer)

        worker.submit(task_1)
        worker.submit(task_2)

        result = worker.execute()
        expected_result = [('peter', 4), ('jenny', 2)]

        self.assertCountEqual(expected_result, result)

    def test_map_reduce_with_less_keys_than_workers(self):
        map_reduce = MapReduce(num_workers=4)
        values = ['key_1', 'key_2', 'key_1']

        expected_result = [('key_1', 2), ('key_2', 1)]

        self.assertCountEqual(expected_result, map_reduce.map_reduce(values, mapper, reducer))

    def test_map_reduce_with_more_keys_than_workers(self):
        map_reduce = MapReduce(num_workers=2)
        values = ['key_1', 'key_2', 'key_1', 'key_2', 'key_3', 'key_4']

        expected_result = [('key_1', 2), ('key_2', 2), ('key_3', 1), ('key_4', 1)]

        self.assertCountEqual(expected_result, map_reduce.map_reduce(values, mapper, reducer))

    def test_map_reduce_with_empty_values(self):
        map_reduce = MapReduce(num_workers=2)
        values = []

        self.assertEqual([], map_reduce.map_reduce(values, mapper, reducer))

    def test_no_interaction_between_map_reduce_runs(self):
        map_reduce = MapReduce(num_workers=4)

        values = ['val_1', 'val_2', 'val_1']
        expected_result = [('val_1', 2), ('val_2', 1)]

        self.assertCountEqual(expected_result, map_reduce.map_reduce(values, mapper, reducer))

        values = ['key_1', 'key_2', 'key_3', 'key_1']
        expected_result = [('key_1', 2), ('key_2', 1), ('key_3', 1)]

        self.assertCountEqual(expected_result, map_reduce.map_reduce(values, mapper, reducer))

    def test_raises_when_non_positive_num_workers(self):
        with self.assertRaises(ValueError):
            MapReduce(num_workers=0)

        with self.assertRaises(ValueError):
            MapReduce(num_workers=-1)

    def test_raises_when_non_callable_map_reduce(self):
        with self.assertRaises(ValueError):
            MapReduce().map_reduce([], 0, lambda x: x)
        with self.assertRaises(ValueError):
            MapReduce().map_reduce([], lambda x: x, 0)

    def test_raises_when_values_not_list(self):
        with self.assertRaises(ValueError):
            MapReduce().map_reduce(0, lambda x: x, lambda x: x)