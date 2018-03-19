    int fastwarp2d_opt(const float * src,
               float * dest_d,
               const int sh[3],
               const int ps[3],
               const float rot,
               const float shear,
               const float scale[2],
               const float stretch_in[2]);
    int fastwarp3d_opt_zxy(const float * src,
                     float * dest_d,
                     const int sh[4],
                     const int ps[4],
                     const float rot,
                     const float shear,
                     const float scale[3],
                     const float stretch_in[4],
                     const float twist_in);

