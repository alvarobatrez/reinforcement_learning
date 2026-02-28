function plot_q_values(Q)
    [rows, cols, actions] = size(Q);

    figure, hold on
    set(gcf, 'Position', [100, 100, 800, 800]);

    colormap('hot')
    cmap = colormap;
    N = size(cmap, 1);
    min_value = min(Q(:));
    max_value = max(Q(:));
    clim([min_value, max_value]);
    clims = clim;

    for i = 1 : rows
        for j = 1 : cols
            for a = 1 : actions
                value = Q(i,j,a);
                if value == 0
                    value = NaN;
                end

                if a == 1
                    x = [j j+0.5 j+1];
                    y = [i i+0.5 i];
                    text_pos = [j+0.5 i+0.25];
                elseif a == 2
                    x = [j+0.5 j+1 j+1];
                    y = [i+0.5 i+1 i];
                    text_pos = [j+0.75 i+0.5];
                elseif a == 3
                    x = [j j+1 j+0.5];
                    y = [i+1 i+1 i+0.5];
                    text_pos = [j+0.5 i+0.75];
                else
                    x = [j j j+0.5];
                    y = [i i+1 i+0.5];
                    text_pos = [j+0.25 i+0.5];
                end
                
                patch(x, rows-y, value, 'EdgeColor','none');

                normalized_value = (value - clims(1)) / (clims(2) - clims(1));
                normalized_value = min(max(normalized_value, 0), 1);
                idx = round(normalized_value * (N - 1)) + 1;
                idx = min(max(idx, 1), N);
                rgb = cmap(idx, :);

                L = 0.299 * rgb(1) + 0.587 * rgb(2) + 0.114 * rgb(3);
                if L > 0.5
                    text_color = [0, 0, 0];
                else
                    text_color = [1, 1, 1];
                end

                text(text_pos(1), rows-text_pos(2), sprintf('%.1f', value), ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                    'FontSize', 7, 'Color', text_color)
            end
        end
    end

    title('Función de Acción-Valor')
    axis off
end
