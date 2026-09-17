% Parâmetros da FFT
fs = 44100;
N = 4096; % Um N comum em processamento de áudio de baixa latência
resolucao_fft = fs / N; % ~10.76 Hz de distância entre cada bin
bins_fft = (0:N/2-1) * resolucao_fft;

% Geração das frequências das notas musicais (Padrão A4 = 440 Hz)
% Calculando notas do A0 (MIDI 21) ao C8 (MIDI 108)
notas_midi = 21:108;
freqs_notas = 440 * 2.^((notas_midi - 69) / 12);

% ----------------- PLOTAGEM -----------------
figure('Position', [100, 100, 1000, 700]);

% Subplot 1: Região Grave (O problema do "Aglomeramento")
subplot(2,1,1);
hold on;
% Bins da FFT (Cinza)
h1 = stem(bins_fft(bins_fft >= 25 & bins_fft <= 100), ...
    ones(1, sum(bins_fft >= 25 & bins_fft <= 100)), ...
    'Color', [0.7 0.7 0.7], 'Marker', 'none', 'LineWidth', 1.5);
% Notas Musicais (Vermelho)
h2 = stem(freqs_notas(freqs_notas >= 25 & freqs_notas <= 100), ...
    1.2 * ones(1, sum(freqs_notas >= 25 & freqs_notas <= 100)), ...
    'r', 'Marker', 'v', 'LineWidth', 2, 'MarkerFaceColor', 'r');

xlim([25 100]);
ylim([0 1.5]);
yticks([]);
title('Região Grave (25 Hz a 100 Hz): Faltam bins da FFT entre as notas (Baixa Resolução)');
xlabel('Frequência (Hz)');
legend([h1, h2], {'Bins da FFT (Resolução Linear ~10.7 Hz)', 'Notas Musicais (Escala Logarítmica)'}, 'Location', 'southeast');
grid on;

% Subplot 2: Região Aguda (O problema do "Gap" / Dados Inúteis)
subplot(2,1,2);
hold on;
% Bins da FFT (Cinza)
stem(bins_fft(bins_fft >= 2000 & bins_fft <= 2500), ...
    ones(1, sum(bins_fft >= 2000 & bins_fft <= 2500)), ...
    'Color', [0.7 0.7 0.7], 'Marker', 'none', 'LineWidth', 1.5);
% Notas Musicais (Vermelho)
stem(freqs_notas(freqs_notas >= 2000 & freqs_notas <= 2500), ...
    1.2 * ones(1, sum(freqs_notas >= 2000 & freqs_notas <= 2500)), ...
    'r', 'Marker', 'v', 'LineWidth', 2, 'MarkerFaceColor', 'r');

xlim([2000 2500]);
ylim([0 1.5]);
yticks([]);
title('Região Aguda (2000 Hz a 2500 Hz): O grande "Gap" de dados inúteis');
xlabel('Frequência (Hz)');

% Destacando o Gap
gap_start = freqs_notas(find(freqs_notas > 2100, 1));
gap_end = freqs_notas(find(freqs_notas > 2100, 2));
gap_end = gap_end(2);
plot([gap_start, gap_end], [0.5, 0.5], 'k<->', 'LineWidth', 1.5);
text((gap_start+gap_end)/2, 0.6, 'Gap: Vários bins da FFT sem nenhuma nota musical', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on;