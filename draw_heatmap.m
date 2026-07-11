function draw_heatmap(V)

figure
data = V;
data(V==0) = NaN;
imagesc(data)
colormap('hot')
clim([min(V(:)), max(V(:))])
set(gca, 'Color', 'w');
alpha(double(~isnan(data)))
axis off
axis equal
title('Función de Valor')