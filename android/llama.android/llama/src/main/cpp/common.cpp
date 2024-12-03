
    bool is_valid_utf8(const char * string) {
        if (!string) {
            return true;
        }

        const unsigned char * bytes = (const unsigned char *)string;
        int num;

        while (*bytes != 0x00) {
            if ((*bytes & 0x80) == 0x00) {
                num = 1;
            } else if ((*bytes & 0xE0) == 0xC0) {
                num = 2;
            } else if ((*bytes & 0xF0) == 0xE0) {
                num = 3;
            } else if ((*bytes & 0xF8) == 0xF0) {
                num = 4;
            } else {
                return false;
            }

            bytes += 1;
            for (int i = 1; i < num; ++i) {
                if ((*bytes & 0xC0) != 0x80) {
                    return false;
                }
                bytes += 1;
            }
        }

        return true;
    }