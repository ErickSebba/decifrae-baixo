% Parâmetros de Áudio
fs = 44100;
duracao = 1;
t = 0:1/fs:duracao-1/fs;

% Acorde Grave: C2 (65.41 Hz), E2 (82.41 Hz), G2 (98.00 Hz)
f_C2 = 65.41;
f_E2 = 82.41;
f_G2 = 98.00;
sinal = sin(2*pi*f_C2*t) + sin(2*pi*f_E2*t) + sin(2*pi*f_G2*t) + 0.05*randn(size(t));

% Configurações da CQT
fmin = 31.0;            % Nota B0 (Limite inferior)
fmax = 4000.0;          % Limite superior (~C8)
bins_por_oitava = 12;   % 12 bins por oitava = 1 bin por semitom musical

% -------------------------------------------------------------------------
% EXECUÇÃO E TRATAMENTO DA CQT
% -------------------------------------------------------------------------
if ~isempty(which('cqt'))
    disp('Usando a função cqt() nativa do Audio Toolbox...');
    
    [cqt_coef, f_cqt] = cqt(sinal, 'SamplingFrequency', fs, ...
                             'BinsPerOctave', bins_por_oitava, ...
                             'FrequencyLimits', [fmin, fmax]);
    
    % Extração de magnitude conforme a estrutura retornada pelo MATLAB
    if isstruct(cqt_coef)
        fn = fieldnames(cqt_coef);
        data_field = fn{1};
        for f_idx = 1:length(fn)
            if isnumeric(cqt_coef(1).(fn{f_idx}))
                data_field = fn{f_idx};
                break;
            end
        end
        
        mag_per_octave = cell(length(cqt_coef), 1);
        for k = 1:length(cqt_coef)
            data = cqt_coef(k).(data_field);
            if ismatrix(data) && size(data, 2) > 1
                mag_per_octave{k} = mean(abs(data), 2);
            else
                mag_per_octave{k} = abs(data(:));
            end
        end
        mag_cqt = vertcat(mag_per_octave{:});
        
    elseif iscell(cqt_coef)
        mag_per_octave = cellfun(@(x) mean(abs(x), 2), cqt_coef, 'UniformOutput', false);
        mag_cqt = vertcat(mag_per_octave{:});
    else
        mag_cqt = mean(abs(cqt_coef), 2);
    end
    
    % --- CORREÇÃO DO TAMANHO (Garantia de Vetores Idênticos) ---
    mag_cqt = mag_cqt(:);
    f_cqt = f_cqt(:);
    
    N_min = min(length(f_cqt), length(mag_cqt));
    f_cqt = f_cqt(1:N_min);
    mag_cqt = mag_cqt(1:N_min);
    
    [f_cqt, idx] = sort(f_cqt);
    mag_cqt = mag_cqt(idx);
    mag_cqt = mag_cqt / max(mag_cqt);
    
else
    disp('Audio Toolbox não encontrado. Executando Reimplementação CQT Kernel...');
    
    % Reimplementação CQT (Algoritmo de Brown & Puckette)
    Q = 1 / (2^(1/bins_por_oitava) - 1);
    K = ceil(bins_por_oitava * log2(fmax / fmin));
    f_cqt = fmin * 2.^((0:K-1) / bins_por_oitava);
    
    N_fft = 2^nextpow2(fs);
    X_fft = fft(sinal, N_fft);
    mag_cqt = zeros(K, 1);
    
    for k = 1:K
        N_k = round(Q * fs / f_cqt(k));
        w = hamming(N_k)';
        s_k = (w / N_k) .* exp(2*pi*1i * f_cqt(k) * (0:N_k-1) / fs);
        
        S_k = fft(s_k, N_fft);
        mag_cqt(k) = abs(sum(X_fft .* conj(S_k)));
    end
    mag_cqt = mag_cqt / max(mag_cqt);
end

% ----------------- PLOTAGEM COMPARATIVA -----------------
figure('Position', [100, 100, 1000, 600]);

% Subplot 1: FFT Clássica (Linear)
N_fft_std = 4096;
X_std = abs(fft(sinal, N_fft_std));
f_std = (0:N_fft_std/2-1) * (fs / N_fft_std);
X_std = X_std(1:N_fft_std/2) / max(X_std(1:N_fft_std/2));

subplot(2,1,1);
plot(f_std, X_std, 'r', 'LineWidth', 1.2);
xlim([30 500]);
title('1. FFT Clássica (Resolução Linear: Picos confusos e vazamento nos graves)');
xlabel('Frequência (Hz)');
ylabel('Magnitude');
grid on;
xline(f_C2, 'b--', 'C2'); xline(f_E2, 'b--', 'E2'); xline(f_G2, 'b--', 'G2');

% Subplot 2: CQT (Logarítmica e Proporcional)
subplot(2,1,2);
stem(f_cqt, mag_cqt, 'b', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
set(gca, 'XScale', 'log');
xlim([30 500]);
title('2. CQT (Resolução Logarítmica: Bins perfeitamente alinhados às notas musicais)');
xlabel('Frequência (Hz) - Escala Logarítmica');
ylabel('Magnitude CQT');
grid on;
xline(f_C2, 'r--', 'C2'); xline(f_E2, 'r--', 'E2'); xline(f_G2, 'r--', 'G2');