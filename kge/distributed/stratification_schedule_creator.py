import random
import numpy as np
from typing import List, Tuple


class StratificationScheduleCreator:
    """
    Creates a non blocking schedule for stratification partitioning
    can only handle num partitions of base 2
    num partitions needs to be 2*num_workers
    """
    def __init__(self, num_partitions, num_workers, randomize_iterations=False, combine_mirror_blocks=False, i_offset=0, j_offset=0):
        self.num_partitions = num_partitions
        self.num_workers = num_workers
        self.randomize_iterations = randomize_iterations
        self.combine_mirror_blocks = combine_mirror_blocks
        self.i_offset = i_offset
        self.j_offset = j_offset
        if self.num_workers*2 > self.num_partitions:
            raise ValueError(f"Can not create strafied schedule for num_workers > num_partitions/2, num_workers={self.num_workers}, num_partitions={self.num_partitions}")

    def create_schedule(self) -> List[List[Tuple[int, int]]]:
        """
        creates non blocking schedule
        Returns:
            list of iterations
            each iteration is a list of blocks (i,j) of size num_workers
        """
        schedule = []
        schedule.extend(self._create_schedule(self.num_partitions))
        if self.randomize_iterations:
            random.shuffle(schedule)
        return schedule

    def _create_schedule(self, n_p, offset=0) -> List[List[Tuple[int, int]]]:
        schedule = []
        if n_p == 2:
            schedule.extend(self._handle_2x2_diagonal(offset=offset))
            return schedule
        else:
            # anti diagonal upper right quadrant
            schedule.extend(
                self._handle_anti_diagonal(
                    int(n_p / 2), x_offset=offset + int(n_p / 2), y_offset=offset
                )
            )
            if not self.combine_mirror_blocks:
                # anti diagonal lower left quadrant
                schedule.extend(
                    self._handle_anti_diagonal(
                        int(n_p / 2), x_offset=offset, y_offset=offset + int(n_p / 2)
                    )
                )
            # both diagonal blocks
            schedule.extend(
                self._concat_schedules(
                    self._create_schedule(int(n_p / 2), offset=offset),
                    self._create_schedule(
                        int(n_p / 2), offset=offset + int(n_p / 2)
                    ),
                )
            )
            return schedule

    #@staticmethod
    def _handle_2x2_diagonal(self, offset=0) -> List[List[Tuple[int, int]]]:
        """
        handle smallest diagonal block
        2x2 with one worker
        Args:
            offset: position in the complete diagonal

        Returns:
            List of iterations for one worker
        """
        schedule = list()
        if not self.combine_mirror_blocks:
            for i in range(2):
                for j in range(2):
                    schedule.append([(i + offset+self.i_offset, j + offset+self.j_offset)])
        else:
            # in combine mirror we only take the upper right block which will be
            #  mirrored later on
            #  only take the lower right diagonal, which will be combined with
            #  diagonal - 1
            schedule.append([(0 + offset, 1 + offset)])
            schedule.append([(1 + offset, 1 + offset)])
        random.shuffle(schedule)
        return schedule

    # @staticmethod
    def _handle_anti_diagonal(self, n_p, x_offset=0, y_offset=0):
        permutation_matrix = np.random.permutation(np.diag(np.ones(n_p, dtype=np.int)))
        schedule = []
        for i in range(int(n_p)):
            iteration = list(zip(*permutation_matrix.nonzero()))
            for j, block in enumerate(iteration):
                block = ((block[0] + i) % (int(n_p)) + x_offset + self.i_offset, block[1] + y_offset + self.j_offset)
                iteration[j] = block
            schedule.append(iteration)
        return schedule

    @staticmethod
    def _concat_schedules(schedule_1, schedule_2):
        schedule = list()
        for iteration_1, iteration_2 in zip(schedule_1, schedule_2):
            iteration_1.extend(iteration_2)
            schedule.append(iteration_1)
        return schedule
