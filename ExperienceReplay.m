classdef ExperienceReplay
    properties
        capacity
        memory
        position = 1
        current_size = 0
    end

    methods
        function obj = ExperienceReplay(capacity)
            obj.capacity = capacity;
            obj.memory = zeros(capacity, 7);
        end

        function obj = insert(obj, transition)
            obj.memory(obj.position, :) = transition;
            
            if obj.current_size < obj.capacity
                obj.current_size = obj.current_size + 1;
            end
            
            obj.position = mod(obj.position, obj.capacity) + 1;
        end

        function batch = sample(obj, batch_size)
            indices = randperm(obj.current_size, batch_size);
            batch = obj.memory(indices, :);
        end

        function result = can_sample(obj, batch_size)
            result = obj.current_size >= batch_size * 10;
        end
    end
end