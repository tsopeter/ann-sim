classdef ContainerInspector
    properties
        conts
    end

    methods
        function ci = ContainerInspector(containers)
            ci.conts = containers
        end

        function x = getbyId(ci, name)
            for p=ci.conts
                x = p;
                if name == p.Id
                    break;
                end
            end
        end
    end
end