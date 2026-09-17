% =========================================================================
% Demonstração: Limitações da FFT para Identificação de Acordes
% =========================================================================

% 1. Parâmetros do Sinal
fs = 44100;             % Taxa de amostragem padrão de áudio (Hz)
duracao = 0.5;          % Duração de meio segundo
t = 0:1/fs:duracao-1/fs; % Vetor de tempo
N = length(t);          % Número de amostras

% 2. Frequências do Acorde de Dó Maior (C4, E4, G4)
% Valores exatos que raramente caem perfeitamente nos "bins" da FFT
f_C4 = 261.63;
f_E4 = 329.63;
f_G4 = 392.00;

% Gerando as ondas com uma leve componente harmônica para realismo
C4 = sin(2*pi*f_C4*t) + 0.3*sin(2*pi*2*f_C4*t);
E4 = sin(2*pi*f_E4*t) + 0.3*sin(2*pi*2*f_E4*t);
G4 = sin(2*pi*f_G4*t) + 0.3*sin(2*pi*2*f_G4*t);

% Acorde final (soma das notas) + ruído branco para simular sinal real
acorde = C4 + E4 + G4;
acorde = acorde + 0.1 * randn(size(t));

% 3. Cálculo da FFT
Y = fft(acorde);
P2 = abs(Y/N);          % Espectro bilateral
P1 = P2(1:floor(N/2)+1); % Espectro unilateral
P1(2:end-1) = 2*P1(2:end-1);

% Eixo de frequências da FFT (Linear)
f_fft = fs*(0:(N/2))/N; 

% 4. Geração dos Gráficos
figure('Name', 'FFT em Acordes Musicais', 'Position', [100, 100, 900, 600]);

% Gráfico 1: O problema dos "Dados Inúteis" (Escala Linear)
subplot(2,1,1);
plot(f_fft, 20*log10(P1 + 1e-6)); % Plot em dB para melhor visualização
title('Espectro Completo da FFT: O Excesso de Dados Inúteis');
xlabel('Frequência (Hz)');
ylabel('Magnitude (dB)');
xlim([0 fs/2]); % Mostra até 22.050 Hz
grid on;
hold on;
% Destacando a região útil para a transcrição
patch([0 2000 2000 0], [-120 -120 10 10], 'g', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
text(2500, -20, '\leftarrow Todo este espaço (> 2kHz) gasta processamento mas não contém notas fundamentais', 'FontSize', 10, 'Color', 'r');
hold off;

% Gráfico 2: O problema do Vazamento Espectral e Resolução
subplot(2,1,2);
plot(f_fft, P1, 'b', 'LineWidth', 1.5);
title('Zoom no Acorde: Vazamento Espectral e "Bordas" Espalhadas');
xlabel('Frequência (Hz)');
ylabel('Magnitude Linear');
xlim([200 900]); % Focando onde as notas estão
grid on;
hold on;

% Linhas indicando onde a energia *deveria* estar exatamente
xline(f_C4, 'r--', 'C4 (261.63 Hz)', 'LabelVerticalAlignment', 'bottom');
xline(f_E4, 'r--', 'E4 (329.63 Hz)', 'LabelVerticalAlignment', 'bottom');
xline(f_G4, 'r--', 'G4 (392.00 Hz)', 'LabelVerticalAlignment', 'bottom');

% Adicionando anotação sobre o vazamento
text(450, max(P1)*0.8, 'Note que as bases dos picos são largas (Vazamento).', 'FontSize', 10);
text(450, max(P1)*0.7, 'A energia da nota espalha para os "bins" vizinhos da FFT.', 'FontSize', 10);
hold off;