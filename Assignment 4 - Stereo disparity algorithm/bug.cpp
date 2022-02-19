#include <iostream>
#include <vector>
#include <string>

class Image
{
public:
    std::vector<unsigned char> image;
    std::string name;
    size_t width;
    size_t height;

    void createEmpty(size_t width, size_t height)
    {
        this->width = width;
        this->height = height;
        this->name = "penismaa";
    }

    void save(const std::string &filename)
    {
        std::cout << "Saved" << std::endl;
    }
};

int main()
{
    Image img;

    img.createEmpty(1234, 5678);

    std::cout << img.width << "x" << img.height << std::endl;

    img.save("kakkapyllypieru");
    img.save("kakkapyllypieru");
    img.save("kakkapyllypieru");
    img.save("kakkapyllypieru");
    img.save("kakkapyllypieru");
    img.save("kakkapyllypieru");
    img.save("kakkapyllypieru");
    img.save("kakkapyllypieru");
    img.save("kakkapyllypieru");

    std::cout << img.width << "x" << img.height << std::endl;
}
