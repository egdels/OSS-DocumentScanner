package docquad;

/**
 * Letterbox transformation: maps a source rectangle into a destination rectangle
 * while preserving aspect ratio, centering with black bars.
 * <p>
 * Immutable. No Android dependencies.
 */
public final class DocQuadLetterbox {

    public final int srcW;
    public final int srcH;
    public final int dstW;
    public final int dstH;

    public final double scale;
    public final double offsetX;
    public final double offsetY;

    private DocQuadLetterbox(int srcW, int srcH, int dstW, int dstH,
                             double scale, double offsetX, double offsetY) {
        this.srcW = srcW;
        this.srcH = srcH;
        this.dstW = dstW;
        this.dstH = dstH;
        this.scale = scale;
        this.offsetX = offsetX;
        this.offsetY = offsetY;
    }

    /**
     * Creates a letterbox mapping from (srcW×srcH) → (dstW×dstH).
     */
    public static DocQuadLetterbox create(int srcW, int srcH, int dstW, int dstH) {
        if (srcW <= 0 || srcH <= 0) throw new IllegalArgumentException("srcW/srcH must be > 0");
        if (dstW <= 0 || dstH <= 0) throw new IllegalArgumentException("dstW/dstH must be > 0");

        double s = Math.min((double) dstW / srcW, (double) dstH / srcH);
        double newW = srcW * s;
        double newH = srcH * s;
        double ox = (dstW - newW) / 2.0;
        double oy = (dstH - newH) / 2.0;
        return new DocQuadLetterbox(srcW, srcH, dstW, dstH, s, ox, oy);
    }

    /**
     * Creates a letterbox mapping from (srcW×srcH) → (256×256).
     */
    public static DocQuadLetterbox create(int srcW, int srcH) {
        return create(srcW, srcH, 256, 256);
    }

    /**
     * Forward: source coordinates → destination (256-space) coordinates.
     */
    public double[] forward(double x, double y) {
        return new double[]{x * scale + offsetX, y * scale + offsetY};
    }

    /**
     * Inverse: destination (256-space) coordinates → source coordinates.
     */
    public double[] inverse(double x, double y) {
        return new double[]{(x - offsetX) / scale, (y - offsetY) / scale};
    }
}
