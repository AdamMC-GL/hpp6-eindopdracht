from mpi4py import MPI
import time


def get_index(start_position, end_position, k):
    for i in range(start_position, end_position+1):
        if i % k == 0 and i != 0:
            return i
    return -1


def find_primes(N, k=2):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    size = N // size

    # when deviding the list, the rank and sizes are used to determine where to start and end in the list
    start_position = comm.Get_rank() * size
    end_position = (comm.Get_rank()+1 * size)
    startTime = time.time()
    zeef = [True for i in range(size)]
    zeef[0] = False

    if rank == 0:
        startTime = time.time()
        while True:
            if k ** 2 > end_position:
                break
            if zeef[k-1]:
                for i in range(k * 2, end_position+1, k):
                    zeef[i-1] = False
            k += 1
        zeef = zeef[start_position:end_position]
    else:
        while True:
            if k ** 2 > end_position:
                break
            index = get_index(start_position, end_position, k)

            if index is not -1 and zeef[index - 1]:
                if index != k:
                    zeef[index - 1] = False
                for i in range(index + k, end_position+1, k):
                    zeef[i - 1] = False

            k += 1
        zeef = zeef[start_position:end_position]

    amount_primes = 0
    for i in range(len(zeef)):
        if zeef[i]:
            amount_primes += 1
    amount_primes = comm.reduce(amount_primes, MPI.SUM, 0)
    if rank == 0:
        duration = time.time() - startTime
        print("Number of primes found: " + str(amount_primes))
        print("Duration: " + str(duration))
    return amount_primes


if __name__ == "__main__":
    assert find_primes(100) == 25, "should be 25"
    assert find_primes(1000) == 168, "should be 168"

    find_primes(100000)


