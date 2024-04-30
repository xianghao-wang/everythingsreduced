module Reducer {
    record Combine {
        type elementType;
        
        proc this(a: elementType, b: elementType): elementType do
            return 0: elementType;
    }

    record Select {
        type containerType;
        type elementType;

        proc this(ref data: containerType, n: int): elementType do
            return 0: elementType;
    }

    record Reducer {
        type elementType;
        type containerType;

        proc doReduce(
            ref data: containerType
            , numElements: int 
            , combiner: Combine(elementType)
            , selecter: Select(containerType, elementType)
            ): elementType {
            return new Combine(new Select(data, 0), new Select(data, 1));
        }
    }
}