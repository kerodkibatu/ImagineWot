using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.PixelFormats;
using static TorchSharp.torch;
using ProgressBar = ShellProgressBar.ProgressBar;
using Image = SixLabors.ImageSharp.Image;
using TorchSharp;
public class ImageLoader
{
    private List<string> imagePaths;
    private int targetImgSize;
    private Tensor loadedImages; // Add a field to store the loaded images

    public ImageLoader(string folderPath, int imgSize)
    {
        imagePaths = new List<string>(Directory.GetFiles(folderPath, "*.*", SearchOption.AllDirectories));
        targetImgSize = imgSize;
        // if the images are already loaded, don't load them again
        if (File.Exists($"./loadedImages{imgSize}.pt"))
        {
            loadedImages = torch.load($"./loadedImages{imgSize}.pt");
        }
        else
        {
            loadedImages = LoadAllImages(); // Load all the images
            // save the loaded images
            torch.save(loadedImages, $"./loadedImages{imgSize}.pt");
        }
    }

    public int Count()
    {
        return imagePaths.Count;
    }

    public Tensor LoadImage(int index)
    {
        if (index < 0 || index >= imagePaths.Count)
        {
            throw new IndexOutOfRangeException($"Index {index} is out of range for dataset with {imagePaths.Count} images.");
        }

        string imagePath = imagePaths[index];
        using (Image<Rgb24> image = Image.Load<Rgb24>(imagePath)) // Use ImageSharp to load the image
        {
            // Resize and Crop
            image.Mutate(x => x.Resize(targetImgSize, targetImgSize)); // ImageSharp's way to resize and crop
            
            // Convert to Tensor
            Tensor tensor = ImageToTensor(image); // Adjust ImageToTensor to accept Image<Rgb24>

            return tensor;
        }
    }

    public IEnumerable<Tensor> LoadImageBatch(int batchSize)
    {

        var indices = Enumerable.Range(0, imagePaths.Count-1).OrderBy(Random.Shared.Next).Take(batchSize);
        foreach (var index in indices)
        {
            yield return loadedImages[index].unsqueeze(0); // Return the pre-loaded image
        }
    }

    private Tensor LoadAllImages()
    {
        List<Tensor> images = [];
        using var progressBar = new ProgressBar(imagePaths.Count, "Loading Images");
        Parallel.For(0, imagePaths.Count, (i) =>
        {
            var tensor = LoadImage(i).unsqueeze(0);
            images.Add(tensor);
            progressBar.Tick($"Loading Images ({images.Count}/{imagePaths.Count})");
        });
        progressBar.Dispose();
        return torch.cat(images,0);
    }

    private Tensor ImageToTensor(Image<Rgb24> image)
    {
        int width = image.Width;
        int height = image.Height;

        var tensor = zeros(new long[] { 3, height, width }, float32);

        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < height; y++)
            {
                Span<Rgb24> pixelRow = accessor.GetRowSpan(y);
                for (int x = 0; x < width; x++)
                {
                    Rgb24 pixel = pixelRow[x];
                    tensor[0, y, x] = pixel.R / 255.0f;
                    tensor[1, y, x] = pixel.G / 255.0f;
                    tensor[2, y, x] = pixel.B / 255.0f;
                }
            }
        });

        return tensor;
    }
}
