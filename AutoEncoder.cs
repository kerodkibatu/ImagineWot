// torchsharp usings
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Layer = TorchSharp.torch.nn.Module<TorchSharp.torch.Tensor, TorchSharp.torch.Tensor>;
public class AutoEncoder : Layer
{
    Sequential encoder;
    Sequential decoder;
    public AutoEncoder(int inputSize, int latentSize, int layersCount) : base("AutoEncoder")
    {
        encoder = Sequential();
        int layerSize = inputSize;
        for (int i = 0; i < layersCount; i++)
        {
            if (i == layersCount - 1)
            {
                encoder.append(Linear(layerSize, latentSize, false));
                encoder.append(ReLU());
            }
            else
            {
                encoder.append(Linear(layerSize, layerSize / 2, false));
                encoder.append(ReLU());
                layerSize /= 2;
            }
        }
        decoder = Sequential();
        layerSize = latentSize;
        for (int i = 0; i < layersCount; i++)
        {
            if (i == layersCount - 1)
            {
                decoder.append(Linear(layerSize, inputSize, false));
                decoder.append(Sigmoid());
            }
            else
            {
                decoder.append(Linear(layerSize, layerSize * 2, false));
                decoder.append(ReLU());
                layerSize *= 2;
            }
        }
        RegisterComponents();
    }
    public Tensor? LastLatent;
    public override Tensor forward(Tensor input)
    {
        var flattened = input.view([input.size(0), -1]);
        var encoded = encoder.forward(flattened);
        LastLatent = encoded;
        var decoded = decoder.forward(encoded);
        return decoded.view([input.size(0), 3, 64, 64]);
    }
    public Tensor Decode(Tensor latent)
    {
        return decoder.forward(latent).view([3, 64, 64]);
    }
}
public class ConvAutoEncoder : Layer
{
    public Sequential Encoder;
    public Sequential Decoder;
    public ConvAutoEncoder(int latentDim = 32, int imageSize = 64) : base("ConvAutoEncoder")
    {
        Encoder = Sequential(
            Conv2d(3, 32, 3, padding: 1),
            ReLU(),
            MaxPool2d(2),
            Conv2d(32, 64, 3, padding: 1),
            ReLU(),
            MaxPool2d(2),
            Conv2d(64, 128, 3, padding: 1),
            ReLU(),
            MaxPool2d(2),
            Conv2d(128, 256, 3, padding: 1),
            ReLU(),
            MaxPool2d(2),
            Flatten(),
            Linear(256 * (imageSize / 16) * (imageSize / 16), latentDim)
        );
        Decoder = Sequential(
            Linear(latentDim, 256 * (imageSize / 16) * (imageSize / 16)),
            ReLU(),
            Unflatten(-1, [256, imageSize / 16, imageSize / 16]),
            Upsample(scale_factor: [2, 2]),
            Conv2d(256, 128, 3, padding: 1),
            ReLU(),
            Upsample(scale_factor: [2, 2]),
            Conv2d(128, 64, 3, padding: 1),
            ReLU(),
            Upsample(scale_factor: [2, 2]),
            Conv2d(64, 32, 3, padding: 1),
            ReLU(),
            Upsample(scale_factor: [2, 2]),
            Conv2d(32, 3, 3, padding: 1),
            Sigmoid()
        );
        RegisterComponents();
    }
    public override Tensor forward(Tensor input)
    {
        var encoded = Encoder.forward(input);
        return Decoder.forward(encoded);
    }
    public Tensor Encode(Tensor input)
    {
        return Encoder.forward(input);
    }
    public Tensor Decode(Tensor latent)
    {
        return Decoder.forward(latent);
    }
}/*
public class ConvolutionalVAE : Layer
{
    public Sequential Encoder;
    public Sequential Decoder;
    public ConvolutionalVAE(int latentDim = 32,
}
*/