import Foundation
@preconcurrency import MLX
import MLXAudioCore
import MLXNN

extension LFM2AudioModel {
    /// Same generation loop as `generateInterleaved`, but it runs on the calling thread and hands
    /// every token to `onToken` the moment it is produced.
    ///
    /// `generateInterleaved` builds its `AsyncThrowingStream` inside the stream's initializer,
    /// which runs synchronously, so the whole reply is generated before the caller receives the
    /// first token. That adds the full generation time to the wait for the first sound.
    ///
    /// There is deliberately no background task or stream here. MLX is not thread-safe, so a
    /// caller that decodes audio on another thread while the model generates could crash. With a
    /// callback the caller does its own decoding on this same thread, one step at a time.
    ///
    /// - Parameters:
    ///   - onPrefillDone: called once the input (text, audio) has been processed and the model is
    ///     ready to produce its first token.
    ///   - onToken: called for each token. For `.text` the token is a scalar; for `.audioOut` it is
    ///     an `(8,)` frame of codebook values (or the end-of-audio frame). Return `false` to stop.
    public func generateInterleavedInline(
        textTokens: MLXArray? = nil,
        audioFeatures: MLXArray? = nil,
        audioCodes: MLXArray? = nil,
        modalities: MLXArray? = nil,
        config genConfig: LFMGenerationConfig = LFMGenerationConfig(),
        onPrefillDone: (() -> Void)? = nil,
        onToken: (MLXArray, LFMModality) -> Bool
    ) {
        let nText = config.interleavedNText
        let nAudio = config.interleavedNAudio

        let (hiddenStates, cache) = prefill(
            textTokens: textTokens, audioFeatures: audioFeatures,
            audioCodes: audioCodes, modalities: modalities
        )
        // Finish the prefill now so `onPrefillDone` marks the real end of that step.
        eval(hiddenStates)
        onPrefillDone?()

        var lastHidden = hiddenStates[0..., (-1)..., 0...]
        var generated = 0
        var modalityLeft = nText
        var textDone = false
        var currentModality = LFMModality.text

        while generated < genConfig.maxNewTokens {
            if Task.isCancelled { return }

            if currentModality == .text {
                let textLogits = lfm.embedTokens.asLinear(lastHidden)[0..., -1, 0...]
                let textToken = sampleTextToken(
                    logits: textLogits, temperature: genConfig.temperature,
                    topK: genConfig.topK
                )
                let tokenId = textToken.item(Int.self)

                if tokenId == lfmImEndToken { break }

                if !onToken(textToken, .text) { return }

                if tokenId == lfmTextEndToken { textDone = true }

                let nextEmb = embedText(MLXArray([Int32(tokenId)]).reshaped(1, 1))
                lastHidden = lfm(inputEmbeddings: nextEmb, cache: cache)

                modalityLeft -= 1
                generated += 1

                if modalityLeft <= 0 || textDone {
                    modalityLeft = nAudio
                    currentModality = .audioOut
                }
            } else {
                let (audioFrame, _) = sampleAudioFrame(
                    hiddenState: lastHidden, audioCache: nil,
                    temperature: genConfig.audioTemperature,
                    topK: genConfig.audioTopK
                )

                if audioFrame[0, 0].item(Int.self) == lfmAudioEOSToken {
                    let eosFrame = MLX.full(audioFrame.shape, values: MLXArray(Int32(lfmAudioEOSToken)), type: Int32.self)
                    if !onToken(eosFrame.squeezed(axis: 0), .audioOut) { return }

                    // Embed EOS back into the model
                    let nextEmb = embedAudioOut(eosFrame).expandedDimensions(axis: 1)
                    lastHidden = lfm(inputEmbeddings: nextEmb, cache: cache)

                    generated += 1
                    currentModality = .text
                    if textDone { break }
                    continue
                }

                if !onToken(audioFrame.squeezed(axis: 0), .audioOut) { return }

                let nextEmb = embedAudioOut(audioFrame).expandedDimensions(axis: 1)
                lastHidden = lfm(inputEmbeddings: nextEmb, cache: cache)

                modalityLeft -= 1
                generated += 1

                if modalityLeft <= 0 && !textDone {
                    modalityLeft = nText
                    currentModality = .text
                }
            }
        }
    }
}
