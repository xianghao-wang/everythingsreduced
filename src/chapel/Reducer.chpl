module Reducer {
    use Shared;
    use GPU;

    proc doReduce(
        type containerType
        , type elementType
        , ref data: containerType // dataset to be reduced
        , numElements: int      // total number of elements
        , identity: elementType // identity in terms of the reduction operator
        ): elementType {
        const DOT_NUM_BLOCKS = min((numElements + TBSIZE - 1) / TBSIZE, 256);
        var blockSum: [0..#DOT_NUM_BLOCKS] elementType;
        const numThreads = TBSIZE * DOT_NUM_BLOCKS;

        @assertOnGpu foreach i in 0..#numThreads {
            var tbSum = createSharedArray(elementType, TBSIZE);
            const localI = i % TBSIZE;
            const blockDimX = TBSIZE;
            tbSum[localI] = identity;

            var j = i;
            while j < numElements {
                tbSum[localI] = tbSum[localI] + (data by j);
                j += numThreads;
            }

            var offset = blockDimX / 2;
            while offset > 0 {
                syncThreads();
                if localI < offset {
                    tbSum[localI] = tbSum[localI] + tbSum[localI+offset];
                }
                offset /= 2;
            }

            if localI == 0 {
                const blockIdxX = i / TBSIZE;
                blockSum[blockIdxX] = tbSum[localI];
            }
        }

        result = + reduce blockSum;

        return result;
    }
}