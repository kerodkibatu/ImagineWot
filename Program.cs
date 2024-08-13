using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using static TorchSharp.torch.nn;
using ConsoleTables;
using Point = SixLabors.ImageSharp.Point;
using ProgressBar = ShellProgressBar.ProgressBar;
using TorchSharp.Modules;

var device = cuda.is_available() ? CUDA : CPU;
int imgSize = 128;

// make sure to set the project working directory to the project root folder
ConvAutoEncoder autoEncoder = (ConvAutoEncoder)new ConvAutoEncoder(32, imgSize)
    .load("models\\model001710.pt");
autoEncoder.to(device);


// Print Model Summary(Params, Layers, etc)
//num_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
Console.WriteLine($"Model Parameters: {autoEncoder.parameters().Where(p => p.requires_grad).Sum(p => p.numel())}");
// recursive function to print model summary table using the console tables library from the state_dict
void PrintModelSummary(Module model)
{
    // get the state_dict
    var state_dict = model.state_dict();
    // create a table
    var table = new ConsoleTable("Layer", "OutputShape", "Parameters");
    // iterate over the state_dict
    foreach (var (key, value) in state_dict)
    {
        // get the expected input shape of the tensor
        var shape = value.shape;

        // get the number of parameters
        var parameters = value.numel();
        // add the row to the table
        table.AddRow(key, string.Join("x", shape.Reverse()), parameters);
    }
    // print the table
    Console.WriteLine(table.ToMinimalString());
}
PrintModelSummary(autoEncoder);

//autoEncoder = (ConvAutoEncoder)autoEncoder.load("models/model300.pt");
Optimizer optimizer = AdamW(autoEncoder.parameters(), 1e-4f);
MSELoss lossFunction = MSELoss(reduction: Reduction.Sum);
ImageLoader Loader = new ImageLoader(@"C:\Users\Kerod\Desktop\ETFood", imgSize);
int batchSize = 16;
int startAt = 1000;
int epochs = 10_000;
int saveEvery = 10;
int ckptToKeep = 2;
int iterations = Loader.Count() / batchSize;
// train
List<float> losses = new();

for (int i = startAt; i < epochs; i++)
{
    var totalLoss = 0f;
    using (var progressBar = new ProgressBar(Loader.Count() / batchSize, "Training"))
    {
        for (int j = 0; j < iterations; j++)
        {
            var image = Loader.LoadImageBatch(batchSize).ToList();
            var target = torch.cat(image, 0);
            target = target.to(device);

            var input = DominantExtractionOverBatch(target);

            optimizer.zero_grad();
            Tensor output = autoEncoder.forward(input);
            Tensor loss = lossFunction.forward(output, target)/batchSize;

            loss.backward();
            totalLoss += loss.item<float>();
            optimizer.step();
            progressBar.Tick(
                $"Epoch {i + 1} ({j}/{Loader.Count() / batchSize}) Loss: {loss.item<float>()}");
            // decrease the learning rate=
        }
        var epochLoss = totalLoss / iterations;
        losses.Add(epochLoss);

        progressBar.Message = $"Epoch {i + 1} completed. Final Loss: {epochLoss} Lr: {optimizer.ParamGroups.First().LearningRate *= 0.99f}";
    }
    if (i % saveEvery == 0)
    {
        // save model
        autoEncoder.save($"models/model{i:000000}.pt");
        // delete old models if more than ckptToKeep
        if (i / saveEvery > ckptToKeep)
        {
            for (int j = 0; j <= i / saveEvery - ckptToKeep; j++)
            {
                File.Delete($"models/model{j * saveEvery:000000}.pt");
            }
        }
        // preview output
        //sample random image
        var img = DominantExtractionOverBatch(Loader.LoadImage(Random.Shared.Next(0, Loader.Count())).unsqueeze(0).to(device));
        Tensor output = autoEncoder.forward(img);

        Preview(img, output, i);
        img.Dispose();
        output.Dispose();
    }
    GC.Collect();
}
Tensor DominantExtractionOverBatch(Tensor batch)
{
    var input = batch.clone();
    for (int k = 0; k < input.size(0); k++)
    {
        var img = input[k];
        var dominant = img.argmax(0).ToInt32();
        for (int l = 0; l < 3; l++)
        {
            img[l] = l == dominant ? 1 : 0;
        }
    }
    return input;
}
// Preview Output using SixLabors.ImageSharp.Drawing
void Preview(Tensor real, Tensor generated, int epoch)
{
    var realImage = TensorToImage(real);
    var generatedImage = TensorToImage(generated);
    using (var image = new Image<Rgb24>(imgSize * 2, imgSize))
    {
        image.Mutate(x =>
        {
            x.DrawImage(realImage, new Point(0, 0), 1f);
            x.DrawImage(generatedImage, new Point(imgSize, 0), 1f);
        });
        image.Save($"outputs/output_at_{epoch+1}.png");
    }
}

Image<Rgb24> TensorToImage(Tensor tensor)
{
    var image = new Image<Rgb24>(imgSize, imgSize);
    var data = tensor.view([3, imgSize, imgSize]).to(DeviceType.CPU).detach();
    for (int i = 0; i < imgSize; i++)
    {
        for (int j = 0; j < imgSize; j++)
        {
            var r = data[0, j, i];
            var g = data[1, j, i];
            var b = data[2, j, i];
            image[i, j] = new Rgb24((byte)(r * 255), (byte)(g * 255), (byte)(b * 255));
        }
    }
    return image;
}