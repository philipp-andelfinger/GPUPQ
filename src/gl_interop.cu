#include <cuda_gl_interop.h>
#include "util.cuh"
#include "SFML/Graphics.hpp"
#include "SFML/Graphics/Image.hpp"

#ifdef A_STAR_REALTIME_VISUALIZATION

#define GL_INTEROP_WIN_WIDTH A_STAR_GRID_X
#define GL_INTEROP_WIN_HEIGHT A_STAR_GRID_Y

static sf::Sprite sprite;
static sf::RenderWindow window(sf::VideoMode(GL_INTEROP_WIN_WIDTH, GL_INTEROP_WIN_HEIGHT), "cuda_gl_interop");
static sf::Texture txture;
cudaSurfaceObject_t bitmap_surface;

void gl_interop_init()
{
  txture.create(GL_INTEROP_WIN_WIDTH, GL_INTEROP_WIN_HEIGHT);

  cudaArray *bitmap_d;

  GLuint gl_tex_handle = txture.getNativeHandle();

  cudaGraphicsResource *cuda_tex_handle;

  cudaGraphicsGLRegisterImage(&cuda_tex_handle, gl_tex_handle, GL_TEXTURE_2D,
                              cudaGraphicsRegisterFlagsNone);
  CudaCheckError();

  cudaGraphicsMapResources(1, &cuda_tex_handle, 0);
  CudaCheckError();

  cudaGraphicsSubResourceGetMappedArray(&bitmap_d, cuda_tex_handle, 0, 0);
  CudaCheckError();

  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;

  resDesc.res.array.array = bitmap_d;
  cudaCreateSurfaceObject(&bitmap_surface, &resDesc);
  CudaCheckError();

  sprite.setTexture(txture);
}

void gl_interop_draw()
{
  window.draw(sprite);
  window.display();
}

#endif
