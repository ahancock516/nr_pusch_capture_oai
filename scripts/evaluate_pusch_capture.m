%% evaluate_pusch_capture.m
%
% Load a PUSCH capture exported from Sionna-RK's nr_pusch_capture plugin
% and evaluate it using MATLAB's 5G Toolbox (nrPUSCHDecode).
%
% Prerequisites:
%   - 5G Toolbox (https://www.mathworks.com/products/5g.html)
%   - .mat files produced by export_to_mat.py
%
% Usage:
%   Run this script or call:
%     evaluate_pusch_capture('path/to/capture_0000.mat')

function results = evaluate_pusch_capture(mat_file)

    if nargin < 1
        mat_file = 'capture_0000.mat';
    end

    fprintf('Loading %s ...\n', mat_file);
    data = load(mat_file);

    rxGrid = data.rxGrid;   % [K x 14] complex double
    chEst  = data.chEst;    % [K x 14] complex double
    oaiLLR = data.llr;      % [N x 1]  double (from OAI default receiver)
    m      = data.meta;      % struct with PHY metadata

    %% ====================================================================
    % Carrier configuration
    % =====================================================================
    carrier = nrCarrierConfig;
    carrier.SubcarrierSpacing  = 30;            % kHz (SCS index 1 = 30 kHz)
    carrier.CyclicPrefix       = 'normal';
    carrier.NSizeGrid          = 106;           % carrier bandwidth in PRBs
    carrier.NStartGrid         = 0;
    carrier.NSlot              = m.slot;
    carrier.NFrame             = m.frame;

    %% ====================================================================
    % PUSCH configuration
    % =====================================================================
    pusch = nrPUSCHConfig;
    pusch.NSizeBWP             = 106;
    pusch.NStartBWP            = m.bwp_start;
    pusch.Modulation           = m.modulation;
    pusch.NumLayers            = m.num_layers;
    pusch.PRBSet               = (m.rb_start : m.rb_start + m.rb_size - 1);
    pusch.SymbolAllocation     = [m.start_symbol, m.num_symbols];
    pusch.TransformPrecoding   = false;         % OAI default for QPSK/QAM

    % DMRS configuration
    pusch.DMRS.DMRSTypeAPosition      = 2;      % pos2 (from gnb config)
    pusch.DMRS.DMRSLength             = 1;      % single-symbol DMRS
    pusch.DMRS.DMRSAdditionalPosition = 1;      % OAI default: 1 additional
    pusch.DMRS.DMRSConfigurationType  = 1;      % Type 1
    pusch.DMRS.NumCDMGroupsWithoutData = 2;     % standard for Type 1
    pusch.DMRS.NIDNSCID               = m.ul_dmrs_scrambling_id;
    pusch.DMRS.NSCID                   = m.scid;

    % Print configuration summary
    fprintf('\n--- Capture Metadata ---\n');
    fprintf('  Frame/Slot:   %d / %d\n', m.frame, m.slot);
    fprintf('  RNTI:         %d\n', m.rnti);
    fprintf('  Modulation:   %s (Qm=%d)\n', m.modulation, m.qam_mod_order);
    fprintf('  Layers:       %d\n', m.num_layers);
    fprintf('  PRBs:         %d (start=%d)\n', m.rb_size, m.rb_start);
    fprintf('  Symbols:      %d..%d (%d total)\n', ...
            m.start_symbol, m.start_symbol + m.num_symbols - 1, m.num_symbols);
    fprintf('  FFT size:     %d\n', m.ofdm_symbol_size);
    fprintf('  nvar:         %d\n', m.nvar);
    fprintf('\n');

    %% ====================================================================
    % Build the full-carrier resource grid
    % =====================================================================
    % Our capture contains only the allocated subcarriers. We need to place
    % them into the full carrier grid for nrPUSCHDecode.
    %
    % rxGrid is [nb_re_per_sym x 14] — only allocated subcarriers,
    % already extracted from the full OFDM symbol by OAI's extract_rbs.

    K_alloc = size(rxGrid, 1);   % allocated subcarriers
    K_full  = carrier.NSizeGrid * 12;  % full carrier subcarriers

    % Place into full grid at the correct PRB offset
    sc_start = m.rb_start * 12 + 1;  % 1-based MATLAB indexing
    sc_end   = sc_start + K_alloc - 1;

    fullRxGrid = zeros(K_full, 14);
    fullChEst  = zeros(K_full, 14);

    for sym = 1:14
        if ~any(isnan(rxGrid(:, sym)))
            fullRxGrid(sc_start:sc_end, sym) = rxGrid(:, sym);
            fullChEst(sc_start:sc_end, sym)  = chEst(:, sym);
        end
    end

    %% ====================================================================
    % Get PUSCH indices and extract symbols
    % =====================================================================
    [puschIndices, puschIndicesInfo] = nrPUSCHIndices(carrier, pusch);

    % Extract received PUSCH symbols using the indices
    rxSymbols = fullRxGrid(puschIndices);

    % Extract channel estimates at PUSCH locations
    hEst = fullChEst(puschIndices);

    % Noise variance estimate
    % OAI's nvar is in a scaled integer format; for MATLAB we use it as-is
    % or estimate from the data. Here we provide a simple estimate.
    if m.nvar > 0
        % OAI nvar is scaled — use a rough conversion or estimate from data
        noiseEst = estimateNoiseVariance(rxSymbols, hEst, m.qam_mod_order);
    else
        noiseEst = 1e-4;
    end
    fprintf('  Noise variance estimate: %.6f\n', noiseEst);

    %% ====================================================================
    % MMSE equalization
    % =====================================================================
    eqSymbols = mmseEqualize(rxSymbols, hEst, noiseEst);

    %% ====================================================================
    % Decode PUSCH using 5G Toolbox
    % =====================================================================
    % nrPUSCHDecode expects equalized symbols [numDataRE x nLayers], NOT
    % the full resource grid. It performs descrambling and demodulation.

    % --- Path A: OAI channel estimates → MMSE equalize → nrPUSCHDecode ---
    % Feed our MMSE-equalized symbols into the toolbox decoder.
    % nrPUSCHDecode signature: nrPUSCHDecode(carrier, pusch, sym, nVar)
    %   sym  = equalized PUSCH symbols [numDataRE x nLayers]
    %   nVar = noise variance scalar
    [matlabLLR_eq, rxSymbols_eq] = nrPUSCHDecode(carrier, pusch, ...
        eqSymbols, noiseEst);

    % --- Path B: MATLAB channel estimation from DMRS → equalize → decode ---
    % Use nrChannelEstimate with DMRS reference signals for proper estimation.
    % The grid needs a 3rd dimension for Rx antennas: [K x N x nRxAnts]
    fullRxGrid3D = reshape(fullRxGrid, K_full, 14, 1);   % single Rx antenna

    % Generate DMRS indices and reference symbols for channel estimation
    dmrsIndices = nrPUSCHDMRSIndices(carrier, pusch);
    dmrsSymbols = nrPUSCHDMRS(carrier, pusch);
    [hEstML, noiseEstML] = nrChannelEstimate(carrier, fullRxGrid3D, ...
        dmrsIndices, dmrsSymbols);

    % Extract and equalize using MATLAB's channel estimate
    rxSymML = fullRxGrid3D(puschIndices);
    hEstML_pusch = hEstML(puschIndices);
    eqSymbols_ml = mmseEqualize(rxSymML, hEstML_pusch, noiseEstML);

    [matlabLLR_full, rxSymbols_full] = nrPUSCHDecode(carrier, pusch, ...
        eqSymbols_ml, noiseEstML);

    %% ====================================================================
    % Compare OAI vs MATLAB LLR outputs
    % =====================================================================
    % nrPUSCHDecode returns a cell array {cw1, cw2, ...}; extract first cw
    matlabLLR_eq   = matlabLLR_eq{1};
    matlabLLR_full = matlabLLR_full{1};

    fprintf('\n--- Results ---\n');
    fprintf('  OAI LLRs:             %d values\n', numel(oaiLLR));
    fprintf('  MATLAB LLRs (OAI-H):  %d values\n', numel(matlabLLR_eq));
    fprintf('  MATLAB LLRs (ML-H):   %d values\n', numel(matlabLLR_full));

    % Truncate to common length for comparison
    N = min([numel(oaiLLR), numel(matlabLLR_eq), numel(matlabLLR_full)]);

    % Normalize for comparison (OAI LLRs are int16-scaled)
    oai_norm = double(oaiLLR(1:N));
    if max(abs(oai_norm)) > 0
        oai_norm = oai_norm / max(abs(oai_norm));
    end
    ml_eq_norm = double(matlabLLR_eq(1:N));
    if max(abs(ml_eq_norm)) > 0
        ml_eq_norm = ml_eq_norm / max(abs(ml_eq_norm));
    end
    ml_full_norm = double(matlabLLR_full(1:N));
    if max(abs(ml_full_norm)) > 0
        ml_full_norm = ml_full_norm / max(abs(ml_full_norm));
    end

    % Sign agreement: percentage of LLRs with the same sign (same hard decision)
    sign_agree_eq   = 100 * mean(sign(oai_norm) == sign(ml_eq_norm));
    sign_agree_full = 100 * mean(sign(oai_norm) == sign(ml_full_norm));

    % Correlation
    corr_eq   = abs(corrcoef_vec(oai_norm, ml_eq_norm));
    corr_full = abs(corrcoef_vec(oai_norm, ml_full_norm));

    fprintf('  Sign agreement (OAI vs OAI-H):    %.1f%%\n', sign_agree_eq);
    fprintf('  Sign agreement (OAI vs ML-H):     %.1f%%\n', sign_agree_full);
    fprintf('  Correlation    (OAI vs OAI-H):    %.4f\n', corr_eq);
    fprintf('  Correlation    (OAI vs ML-H):     %.4f\n', corr_full);

    % Compute EVM on equalized symbols
    evmObj = comm.EVM('ReferenceSignalSource', 'Estimated from reference constellation');
    evmPercent = evmObj(eqSymbols);
    fprintf('  EVM (equalized):      %.2f%%\n', evmPercent);

    %% ====================================================================
    % Plotting
    % =====================================================================
    figure('Position', [100 100 1800 1200], 'Name', ...
           sprintf('PUSCH Capture #%d — Frame %d Slot %d', ...
                   m.capture_idx, m.frame, m.slot));

    % (1) Constellation
    subplot(3,3,1);
    plot(real(eqSymbols), imag(eqSymbols), '.', 'MarkerSize', 3, 'Color', [0.3 0.5 0.8]);
    hold on;
    % Generate all possible bit patterns to get reference constellation points
    nBits = m.qam_mod_order;
    nPts = 2^nBits;
    allBits = int8(de2bi(0:nPts-1, nBits, 'left-msb')');
    refPts = nrSymbolModulate(allBits(:), m.modulation);
    plot(real(refPts), imag(refPts), 'r+', 'MarkerSize', 10, 'LineWidth', 2);
    hold off;
    grid on; axis equal;
    title(sprintf('Constellation (%s)', m.modulation));
    xlabel('I'); ylabel('Q');

    % (2) Resource grid power
    subplot(3,3,2);
    imagesc(0:13, 0:K_full-1, 20*log10(abs(fullRxGrid) + 1e-6));
    colorbar; colormap(gca, 'parula');
    xlabel('OFDM Symbol'); ylabel('Subcarrier');
    title('|Y(k,l)| [dB]');
    set(gca, 'YDir', 'normal');

    % (3) Channel estimate magnitude
    subplot(3,3,3);
    imagesc(0:13, 0:K_full-1, 20*log10(abs(fullChEst) + 1e-6));
    colorbar; colormap(gca, 'hot');
    xlabel('OFDM Symbol'); ylabel('Subcarrier');
    title('|H(k,l)| [dB]');
    set(gca, 'YDir', 'normal');

    % (4) LLR comparison scatter
    subplot(3,3,4);
    scatter(oai_norm, ml_eq_norm, 3, 'filled', 'MarkerFaceAlpha', 0.3);
    hold on;
    plot([-1 1], [-1 1], 'r--', 'LineWidth', 1.5);
    hold off;
    grid on;
    xlabel('OAI LLR (normalized)');
    ylabel('MATLAB LLR w/ OAI ChEst (normalized)');
    title(sprintf('OAI vs MATLAB-OAI\\_H (corr=%.3f)', corr_eq));
    axis([-1.1 1.1 -1.1 1.1]);

    % (5) LLR histograms overlay
    subplot(3,3,5);
    edges = linspace(-1, 1, 80);
    histogram(oai_norm, edges, 'FaceAlpha', 0.5, 'FaceColor', 'b', ...
              'Normalization', 'probability', 'DisplayName', 'OAI');
    hold on;
    histogram(ml_eq_norm, edges, 'FaceAlpha', 0.5, 'FaceColor', 'r', ...
              'Normalization', 'probability', 'DisplayName', 'MATLAB');
    hold off;
    legend; grid on;
    xlabel('Normalized LLR'); ylabel('Probability');
    title('LLR Distribution');

    % (6) Channel frequency response
    subplot(3,3,6);
    % Find a DMRS symbol to plot
    dmrs_idx = find(m.dmrs_symbols > 0, 1, 'first');
    if ~isempty(dmrs_idx)
        h_dmrs = fullChEst(sc_start:sc_end, dmrs_idx);
        plot(0:K_alloc-1, 20*log10(abs(h_dmrs) + 1e-6), 'r-', 'LineWidth', 1.2);
        hold on;
        % Also plot a data symbol for comparison
        data_idx = find(m.dmrs_symbols == 0);
        data_idx = data_idx(data_idx >= m.start_symbol+1 & ...
                            data_idx <= m.start_symbol+m.num_symbols);
        if ~isempty(data_idx)
            h_data = fullChEst(sc_start:sc_end, data_idx(1));
            plot(0:K_alloc-1, 20*log10(abs(h_data) + 1e-6), 'b-', ...
                 'LineWidth', 0.8, 'Color', [0.3 0.5 0.8 0.5]);
        end
        hold off;
        legend('DMRS symbol', 'Data symbol');
    end
    grid on;
    xlabel('Subcarrier'); ylabel('|H(k)| [dB]');
    title('Channel Frequency Response');

    % =====================================================================
    % Row 3: DMRS spectrograms
    % =====================================================================
    % Generate expected DMRS symbols and locate them on the resource grid
    dmrsIndicesPlot = nrPUSCHDMRSIndices(carrier, pusch);
    dmrsSymbolsRef  = nrPUSCHDMRS(carrier, pusch);

    % Build a grid showing DMRS locations (NaN where no DMRS)
    dmrsGrid = nan(K_full, 14);
    dmrsGrid(dmrsIndicesPlot) = dmrsSymbolsRef;

    % Extract received symbols at DMRS positions
    rxDmrs = fullRxGrid(dmrsIndicesPlot);

    % DMRS symbol indices within the slot (0-based)
    dmrsSym0 = find(m.dmrs_symbols > 0) - 1;   % 0-based for labeling
    dmrsSym1 = find(m.dmrs_symbols > 0);        % 1-based for indexing
    nDmrs = numel(dmrsSym1);

    % (7) DMRS received power spectrogram — magnitude per DMRS symbol
    subplot(3,3,7);
    if nDmrs > 0
        dmrsPowerGrid = nan(K_alloc, nDmrs);
        for di = 1:nDmrs
            sym1 = dmrsSym1(di);
            col = fullRxGrid(sc_start:sc_end, sym1);
            dmrsPowerGrid(:, di) = 20*log10(abs(col) + 1e-6);
        end
        imagesc(1:nDmrs, 0:K_alloc-1, dmrsPowerGrid);
        colorbar; colormap(gca, 'parula');
        set(gca, 'YDir', 'normal');
        set(gca, 'XTick', 1:nDmrs, 'XTickLabel', ...
            arrayfun(@(s) sprintf('Sym %d', s), dmrsSym0, 'UniformOutput', false));
    end
    xlabel('DMRS Symbol'); ylabel('Subcarrier');
    title('DMRS Received Power |Y_{DMRS}(k)| [dB]');

    % (8) DMRS constellation — received vs expected
    subplot(3,3,8);
    if nDmrs > 0
        % Equalize DMRS symbols using OAI channel estimates
        hDmrs = fullChEst(dmrsIndicesPlot);
        eqDmrs = conj(hDmrs) .* rxDmrs ./ (abs(hDmrs).^2 + noiseEst);

        scatter(real(eqDmrs), imag(eqDmrs), 8, 'b', 'filled', ...
                'MarkerFaceAlpha', 0.4, 'DisplayName', 'Rx (equalized)');
        hold on;
        % Expected DMRS constellation (BPSK-like for Type 1)
        scatter(real(dmrsSymbolsRef), imag(dmrsSymbolsRef), 40, 'r', '+', ...
                'LineWidth', 2, 'DisplayName', 'Expected');
        hold off;
        legend('Location', 'best'); grid on; axis equal;
        lim_d = max(abs([real(eqDmrs); imag(eqDmrs)])) * 1.3;
        if lim_d > 0
            axis([-lim_d lim_d -lim_d lim_d]);
        end
    end
    xlabel('I'); ylabel('Q');
    title('DMRS Constellation (Rx vs Ref)');

    % (9) DMRS phase & magnitude per symbol — error analysis
    subplot(3,3,9);
    if nDmrs > 0
        % Per-DMRS-symbol phase error and magnitude ratio
        dmrsPhaseErr = nan(K_alloc, nDmrs);
        dmrsMagRatio = nan(K_alloc, nDmrs);

        nDmrsPerSym = numel(dmrsSymbolsRef) / nDmrs;
        for di = 1:nDmrs
            sym1 = dmrsSym1(di);
            rxCol = fullRxGrid(sc_start:sc_end, sym1);
            hCol  = fullChEst(sc_start:sc_end, sym1);

            % Equalize and compare to expected DMRS
            eqCol = conj(hCol) .* rxCol ./ (abs(hCol).^2 + noiseEst);
            idx_range = (di-1)*nDmrsPerSym+1 : di*nDmrsPerSym;
            refCol = nan(K_alloc, 1);

            % Map DMRS REs back onto allocated subcarrier positions
            % DMRS Type 1: every other subcarrier is a pilot
            dmrsMask = false(K_alloc, 1);
            for ii = idx_range
                % Convert global DMRS index to subcarrier within allocation
                globalSC = mod(dmrsIndicesPlot(ii)-1, K_full);
                localSC = globalSC - (sc_start - 1);
                if localSC >= 0 && localSC < K_alloc
                    dmrsMask(localSC+1) = true;
                    refCol(localSC+1) = dmrsSymbolsRef(ii);
                end
            end

            % Phase error at DMRS subcarriers
            eqDmrsSC = eqCol(dmrsMask);
            refDmrsSC = refCol(dmrsMask);
            phaseErr = angle(eqDmrsSC ./ refDmrsSC) * (180/pi);
            dmrsPhaseErr(dmrsMask, di) = phaseErr;
        end

        % Plot phase error per DMRS symbol as overlaid lines
        hold on;
        colors = lines(nDmrs);
        for di = 1:nDmrs
            validMask = ~isnan(dmrsPhaseErr(:, di));
            scPlot = find(validMask) - 1;
            plot(scPlot, dmrsPhaseErr(validMask, di), '-', ...
                 'Color', colors(di,:), 'LineWidth', 1.0, ...
                 'DisplayName', sprintf('Sym %d', dmrsSym0(di)));
        end
        hold off;
        legend('Location', 'best', 'FontSize', 7);
        grid on;
        ylabel('Phase Error [deg]');
        xlabel('Subcarrier');
        title('DMRS Phase Error (Eq vs Ref)');
    end

    sgtitle(sprintf('PUSCH Capture #%d — Frame %d, Slot %d, RNTI %d, %s, %d PRBs', ...
            m.capture_idx, m.frame, m.slot, m.rnti, m.modulation, m.rb_size), ...
            'FontWeight', 'bold');

    % Save figure
    [fdir, fname] = fileparts(mat_file);
    figpath = fullfile(fdir, [fname '_analysis.png']);
    exportgraphics(gcf, figpath, 'Resolution', 150);
    fprintf('\n  Figure saved: %s\n', figpath);

    % Return results
    results.evm          = evmPercent;
    results.sign_agree   = sign_agree_eq;
    results.correlation  = corr_eq;
    results.noiseEst     = noiseEst;
    results.oaiLLR       = oaiLLR;
    results.matlabLLR_eq = matlabLLR_eq;
    results.matlabLLR_full = matlabLLR_full;
    results.eqSymbols    = eqSymbols;
end

%% Helper functions

function eq = mmseEqualize(rxSym, hEst, noiseVar)
    % MMSE equalization: eq = conj(H) .* rx / (|H|^2 + sigma^2)
    eq = conj(hEst) .* rxSym ./ (abs(hEst).^2 + noiseVar);
end

function nv = estimateNoiseVariance(rxSym, hEst, Qm)
    % Estimate noise variance from error between received and reconstructed
    % signal after hard demodulation
    eq = conj(hEst) .* rxSym ./ (abs(hEst).^2 + 1e-6);
    modStr = getModString(Qm);
    hardBits = nrSymbolDemodulate(eq, modStr, 'DecisionType', 'hard');
    refSym = nrSymbolModulate(hardBits, modStr);
    noise = eq - refSym;
    nv = mean(abs(noise).^2);
end

function s = getModString(Qm)
    switch Qm
        case 2, s = 'QPSK';
        case 4, s = '16QAM';
        case 6, s = '64QAM';
        case 8, s = '256QAM';
        otherwise, s = 'QPSK';
    end
end

function r = corrcoef_vec(a, b)
    % Pearson correlation between two vectors
    a = a(:); b = b(:);
    a = a - mean(a); b = b - mean(b);
    r = (a' * b) / (norm(a) * norm(b) + eps);
end
