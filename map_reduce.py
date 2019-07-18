"""This module defines classes for a simple MapReduce framework"""
import functools

from collections import defaultdict


class Task:
    """Represents a task to be executed.

    A task aggregates all the values for the given key according to a reducer.
    """

    def __init__(self, key, values, reduce):
        """Initializes the task.

        Args:
            key: (str), The key
            values: (list), List of values belonging to the key
            reduce: (callable), The reducer function
        """
        self._key = key
        self._values = values
        self._reduce = reduce

    def run(self):
        """Aggregates the values for the given key.

        Returns: tuple, (key, aggregated_value)

        """
        return self._key, functools.reduce(self._reduce, self._values)


class Worker:
    """Represents a worker executing the actual tasks."""

    def __init__(self):
        """Initializes the worker"""
        self._tasks = []

    def submit(self, task):
        """Submits a task for a later execution.

        Args:
            task: (Task), Task to be executed
        """
        self._tasks.append(task)

    def execute(self):
        """Executes all tasks and returns their combined result

        Returns: (list), Combined result of all tasks
        """
        return [task.run() for task in self._tasks]


class MapReduce:
    """Represents MapReduce framework."""

    def __init__(self, num_workers=1):
        """Initializes MapReduce

        Args:
            num_workers: (int, positive), Number of workers
        """
        if num_workers <= 0:
            raise ValueError('num_workers should be positive')
        self._num_workers = num_workers

    def map_reduce(self, values, map, reduce):
        """Performs map-reduce procedure.

        Args:
            values: (list), Values to process
            map: (callable), Mapper
            reduce: (reduce), Reducer

        Returns: (list), List of tuples of form (key, value) where value is the aggregated value for the key.
        """
        if not type(values) is list:
            raise ValueError('values should be a list')
        if not callable(map):
            raise ValueError('map should be a callable')
        if not callable(reduce):
            raise ValueError('reduce should be a callable')

        workers = self._prepare_tasks(values, map, reduce)
        return self._execute_tasks(workers)

    def _prepare_tasks(self, values, map, reduce):
        mapped = [map(value) for value in values]

        task_inputs = defaultdict(list)

        for key, value in mapped:
            # [('peter', 1), ('jenny', 1), ('peter', 1)] -> {'peter': [1, 1], 'jenny': [1]}
            task_inputs[key].append(value)

        workers = [Worker() for _ in range(self._num_workers)]

        for i, (key, value) in enumerate(task_inputs.items()):
            # distribute tasks between workers circularly
            workers[i % self._num_workers].submit(Task(key, value, reduce))

        return workers

    def _execute_tasks(self, workers):
        result = []
        for worker in workers:
            result.extend(worker.execute())
        return result


if __name__ == '__main__':
    # Average word length calculation for the given text
    text = """But I must explain to you how all this mistaken idea of denouncing pleasure and praising pain was born and I will give you a complete account of the system, and expound the actual teachings of the great explorer of the truth, the master-builder of human happiness. No one rejects, dislikes, or avoids pleasure itself, because it is pleasure, but because those who do not know how to pursue pleasure rationally encounter consequences that are extremely painful. Nor again is there anyone who loves or pursues or desires to obtain pain of itself, because it is pain, but because occasionally circumstances occur in which toil and pain can procure him some great pleasure. To take a trivial example, which of us ever undertakes laborious physical exercise, except to obtain some advantage from it? But who has any right to find fault with a man who chooses to enjoy a pleasure that has no annoying consequences, or one who avoids a pain that produces no resultant pleasure? On the other hand, we denounce with righteous indignation and dislike men who are so beguiled and demoralized by the charms of pleasure of the moment, so blinded by desire, that they cannot foresee the pain and trouble that are bound to ensue; and equal blame belongs to those who fail in their duty through weakness of will, which is the same as saying through shrinking from toil and pain. These cases are perfectly simple and easy to distinguish. In a free hour, when our power of choice is untrammelled and when nothing prevents our being able to do what we like best, every pleasure is to be welcomed and every pain avoided. But in certain circumstances and owing to the claims of duty or the obligations of business is will frequently occur that pleasures have to be repudiated and annoyances accepted. The wise man therefore always holds in these matters to this principle of selection: he rejects pleasures to secure other greater pleasures, or else he endures pains to avoid worse pains."""

    map_reduce = MapReduce(num_workers=4)
    # clean the text
    text = text.replace('.:,;?', '')
    articles = {'a', 'an', 'the'}
    text = ' '.join(word for word in text.split(' ') if word not in articles)

    # calculate the total length of each unique word in the text
    # e.g. for text = 'word1 word2 word1' it would be (('word1', 10), ('word2', 5))
    total_char_count_by_word = map_reduce.map_reduce(
        text.split(' '),
        lambda s: (s, len(s)),
        lambda x, y: x + y)

    # At this point, it was not completely clear to me, what's actually meant by average word length:
    # if it's the average length of unique words or the average length of all words.
    # I calculated the average length of all words.
    avg_word_length = sum(x[1] for x in total_char_count_by_word) / len(text.split(' '))

    print(f'The average word length is: {avg_word_length:.1f}')
